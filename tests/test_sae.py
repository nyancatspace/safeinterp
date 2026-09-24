import torch

from safeinterp.sae import SAE, SAEConfig


def test_topk_l0_and_shapes():
    sae = SAE(SAEConfig(d_in=16, d_sae=64, kind="topk", k=4))
    x = torch.randn(10, 16) * 7
    sae.init_from_data(x)
    z = sae.encode(x)
    assert z.shape == (10, 64)
    assert ((z > 0).sum(-1) <= 4).all()
    assert sae(x).shape == x.shape


def test_act_scale_normalizes_norm():
    sae = SAE(SAEConfig(d_in=16, d_sae=32))
    x = torch.randn(1000, 16) * 50
    sae.init_from_data(x)
    assert abs((x * sae.act_scale).pow(2).sum(-1).mean().item() - 16) < 1e-2


def test_training_reduces_loss_and_keeps_unit_decoder():
    torch.manual_seed(0)
    # Data built from 20 sparse ground-truth directions.
    dirs = torch.randn(20, 16)
    codes = (torch.rand(4096, 20) < 0.1).float() * torch.rand(4096, 20) * 3
    x = codes @ dirs
    for kind in ("topk", "relu"):
        sae = SAE(SAEConfig(d_in=16, d_sae=64, kind=kind, k=4, l1_coeff=0.05))
        sae.init_from_data(x)
        opt = torch.optim.Adam(sae.parameters(), lr=3e-3)
        first = sae.loss(x)["fvu"].item()
        for _ in range(300):
            out = sae.loss(x)
            opt.zero_grad()
            out["loss"].backward()
            sae.remove_parallel_grad()
            opt.step()
            sae.normalize_decoder()
        assert sae.loss(x)["fvu"].item() < 0.5 * first, kind
        assert torch.allclose(sae.W_dec.norm(dim=1), torch.ones(64), atol=1e-5)


def test_save_load_roundtrip(tmp_path):
    sae = SAE(SAEConfig(d_in=8, d_sae=16, k=4, layer=3, hook="mlp_out"))
    sae.init_from_data(torch.randn(50, 8))
    sae.save(tmp_path / "s")
    loaded = SAE.load(tmp_path / "s")
    x = torch.randn(5, 8)
    assert loaded.cfg.layer == 3 and loaded.cfg.hook == "mlp_out"
    assert torch.allclose(loaded(x), sae(x))
