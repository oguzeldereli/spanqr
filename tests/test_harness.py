"""Self-tests of the harness: generators build what they claim, the metrics catch planted defects, and the baselines
reproduce the spec's history (Householder passes; fp32 Cholesky-QR2 fails at condition 1e4+ and on zero X).
Run: pytest -q tests/ (a few seconds on a GPU)."""
import pytest
import torch

from qrbench import gen, metrics, methods

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
K, R = 512, 128


def case(name):
    return next(gen.make_cases(K, R, [name], (0,)))


def check(method, name):
    c = case(name)
    ref = metrics.reference(c.X)
    hres, _ = metrics.run_checked(methods.get("householder"), c, ref, check_sync=False)
    res, Q = metrics.run_checked(methods.get(method), c, ref)
    return res, res.failures(span_ref=hres.span_n), Q


def test_mixed_singular_values():
    S, _ = metrics.reference(case("mixed_c4").X)
    want = torch.logspace(0, -4, R, dtype=torch.float64)
    assert torch.allclose(S, want, rtol=1e-4)


def test_mixed_is_mixed_on_both_sides():
    """not column-scaled: the column norms of a mixed matrix are not graded like its singular values"""
    n = case("mixed_c6").X.norm(dim=0)
    assert n.max() / n.min() < 10


def test_cases_deterministic():
    a, b = case("ttf_cu4_t1e-2"), case("ttf_cu4_t1e-2")
    assert torch.equal(a.X, b.X) and torch.equal(a.ctx["V"], b.ctx["V"])


def test_ttf_structure():
    c = case("ttf_cu4_t1e-2")
    V = c.ctx["V"].double()
    assert (V.T @ V - torch.eye(R, dtype=torch.float64, device=V.device)).abs().max() < 1e-6
    S, _ = metrics.reference(c.X)
    assert S[0] / S[-1] > 1e5          # cond(B) = 1e8 tilted by P


@pytest.mark.parametrize("name", [n for n in gen.case_names(K, R)])
def test_householder_passes_everything(name):
    res, fails, _ = check("householder", name)
    assert not fails, fails


@pytest.mark.parametrize("name", ["mixed_c6", "mixed_c8", "zero", "rank_half", "dup_cols"])
def test_cholqr2_fp32_fails(name):
    _, fails, _ = check("cholqr2_fp32", name)
    assert fails


def test_cholqr2_fp32_fine_when_well_conditioned():
    _, fails, _ = check("cholqr2_fp32", "mixed_c2")
    assert not fails, fails


def _planted(defect):
    def fn(X):
        Q = torch.linalg.qr(X)[0].contiguous()
        return defect(Q)
    return fn


def _run_planted(fn, name="mixed_c4"):
    c = case(name)
    ref = metrics.reference(c.X)
    hres, _ = metrics.run_checked(methods.get("householder"), c, ref, check_sync=False)
    res, _ = metrics.run_checked(fn, c, ref)
    return res, res.failures(span_ref=hres.span_n)


def test_detects_zero_column():
    res, fails = _run_planted(_planted(lambda Q: torch.cat([Q[:, :-1], torch.zeros_like(Q[:, :1])], 1)))
    assert res.bad_cols == 1 and fails


def test_detects_nan():
    res, fails = _run_planted(_planted(lambda Q: Q.index_fill(1, torch.tensor([3], device=Q.device), float("nan"))))
    assert not res.finite and fails


def test_detects_wrong_span():
    """an orthonormal Q of the wrong span (a strong direction swapped for a random orthogonal one)"""
    def defect(Q):
        Z = torch.randn(Q.shape[0], 1, device=Q.device, generator=torch.Generator(Q.device).manual_seed(0))
        Z = Z - Q @ (Q.T @ Z); Z = Z - Q @ (Q.T @ Z)
        return torch.cat([Q[:, 1:], Z / Z.norm()], 1).contiguous()
    res, fails = _run_planted(_planted(defect))
    assert res.orth < 1e-6 and res.span_raw > 0.1 and fails


def test_detects_loss_of_orthogonality():
    res, fails = _run_planted(_planted(lambda Q: (Q + 1e-5 * Q[:, :1]).contiguous()))
    assert res.orth > 1e-6 and fails


def test_detects_host_sync():
    def fn(X):
        Q = torch.linalg.qr(X)[0]
        return Q * (1.0 if float(Q.abs().max()) > 0 else 1.0)    # .item()-style sync
    res, fails = _run_planted(fn)
    assert not res.sync_free and "host sync" in fails


def test_detects_nondeterminism():
    state = [0]
    def fn(X):
        state[0] += 1
        Q = torch.linalg.qr(X)[0].contiguous()
        return Q if state[0] % 2 else -Q
    res, fails = _run_planted(fn)
    assert not res.deterministic and fails


def test_detects_input_modification():
    def fn(X):
        X.mul_(2)
        return torch.linalg.qr(X)[0]
    res, fails = _run_planted(fn)
    assert not res.input_intact and fails


def test_crash_is_reported():
    def fn(X):
        raise ValueError("boom")
    res, fails = _run_planted(fn)
    assert res.error and fails
