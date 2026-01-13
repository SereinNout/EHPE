import os
import yaml
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from ehpe.models.ehpe import EHPE
from ehpe.losses import MPJPELoss, EdgeStabilityLoss
from ehpe.utils.seed import set_seed
from ehpe.datasets.interhand2p6m import InterHand2p6M

def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def freeze_tw(model: EHPE):
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.tw.parameters():
        p.requires_grad = False

def main(cfg_path="configs/default.yaml", tw_ckpt=None):
    cfg = load_cfg(cfg_path)
    set_seed(cfg["seed"])

    device = torch.device(cfg["device"])
    model = EHPE(cfg).to(device)

    if tw_ckpt is not None:
        ck = torch.load(tw_ckpt, map_location="cpu")
        model.load_state_dict(ck["model"], strict=False)

    freeze_tw(model)

    ds = InterHand2p6M(cfg, split="train")
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                    num_workers=cfg["train"]["num_workers"], pin_memory=True)

    lp = MPJPELoss()
    le = EdgeStabilityLoss(mode="l2")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg["train"]["pg"]["lr"], weight_decay=cfg["train"]["pg"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=cfg["train"]["pg"]["lr_milestones"], gamma=cfg["train"]["pg"]["lr_gamma"])

    scaler = GradScaler(enabled=cfg["train"]["amp"])

    model.train()
    for epoch in range(cfg["train"]["pg"]["epochs"]):
        for it, batch in enumerate(dl):
            img = batch["image"].to(device, non_blocking=True)
            gt_xyz = batch["xyz"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=cfg["train"]["amp"]):
                out = model(img)
                pred = out["pg_pred"]
                alphas = out["alphas"]
                loss_p = lp(pred, gt_xyz)
                loss_e = le(alphas)
                total = cfg["loss"]["pg"]["lambda_p"] * loss_p + cfg["loss"]["pg"]["lambda_e"] * loss_e

            scaler.scale(total).backward()
            scaler.step(opt)
            scaler.update()

            if it % cfg["logging"]["log_interval"] == 0:
                lr = opt.param_groups[0]["lr"]
                print(f"[PG] epoch {epoch} it {it} lr {lr:.2e} loss {total.item():.4f} (P {loss_p.item():.4f} E {loss_e.item():.4f})")

        scheduler.step()
        os.makedirs(cfg["logging"]["save_dir"], exist_ok=True)
        torch.save({"model": model.state_dict(), "cfg": cfg}, os.path.join(cfg["logging"]["save_dir"], f"pg_epoch{epoch}.pt"))

if __name__ == "__main__":
    main()
