import os
import yaml
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from ehpe.models.ehpe import EHPE
from ehpe.losses import HeatmapMSELoss, Euclidean3DLoss, L1Regularizer
from ehpe.utils.seed import set_seed
from ehpe.datasets.interhand2p6m import InterHand2p6M

def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main(cfg_path="configs/default.yaml"):
    cfg = load_cfg(cfg_path)
    set_seed(cfg["seed"])

    device = torch.device(cfg["device"])
    model = EHPE(cfg).to(device)

    ds = InterHand2p6M(cfg, split="train")
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                    num_workers=cfg["train"]["num_workers"], pin_memory=True)

    hm_loss = HeatmapMSELoss()
    ed_loss = Euclidean3DLoss()
    l1reg = L1Regularizer()

    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["tw"]["lr"], weight_decay=cfg["train"]["tw"]["weight_decay"])
    scaler = GradScaler(enabled=cfg["train"]["amp"])

    model.train()
    for epoch in range(cfg["train"]["tw"]["epochs"]):
        for it, batch in enumerate(dl):
            img = batch["image"].to(device, non_blocking=True)
            gt_tw_heat = batch["tw_heatmap"].to(device, non_blocking=True)
            gt_tw_xyz = batch["tw_xyz"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=cfg["train"]["amp"]):
                out = model(img)
                pred_heat = out["tw_heat2d"]
                pred_xyz = out["tw_coords_25d"]

                loss_h = hm_loss(pred_heat, gt_tw_heat)
                loss_ed = ed_loss(pred_xyz, gt_tw_xyz)

                loss_r = l1reg([p for p in model.parameters() if p.requires_grad]) * 1e-8
                total = cfg["loss"]["tw"]["lambda_h"] * loss_h +                         cfg["loss"]["tw"]["lambda_ed"] * loss_ed +                         cfg["loss"]["tw"]["lambda_r"] * loss_r

            scaler.scale(total).backward()
            scaler.step(opt)
            scaler.update()

            if it % cfg["logging"]["log_interval"] == 0:
                print(f"[TW] epoch {epoch} it {it} loss {total.item():.4f} (H {loss_h.item():.4f} ED {loss_ed.item():.4f})")

        os.makedirs(cfg["logging"]["save_dir"], exist_ok=True)
        torch.save({"model": model.state_dict(), "cfg": cfg}, os.path.join(cfg["logging"]["save_dir"], f"tw_epoch{epoch}.pt"))

if __name__ == "__main__":
    main()
