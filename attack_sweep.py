# attack_sweep.py
import os
import numpy as np
import torch
import torch.nn.functional as F
import utils
from tqdm import trange

def pgd_attack(model, x, y, eps_norm, alpha_norm, iters, device="cpu", rand_start=True):
    x_nat = x.clone().detach().to(device)
    if rand_start:
        x_adv = x_nat + torch.empty_like(x_nat).uniform_(-1,1) * eps_norm
        x_adv = x_adv.detach()
    else:
        x_adv = x_nat.clone().detach()

    for _ in range(iters):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y.to(device))
        loss.backward()
        grad = x_adv.grad.detach()
        x_adv = x_adv + alpha_norm * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x_nat + eps_norm), x_nat - eps_norm).detach()
    return x_adv.detach()

def fgsm_attack(model, x, y, eps_norm, device="cpu"):
    x_adv = x.clone().detach().to(device).requires_grad_(True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y.to(device))
    loss.backward()
    grad = x_adv.grad.detach()
    x_adv = x_adv + (eps_norm * grad.sign())
    return x_adv.detach()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    here = os.path.dirname(os.path.abspath(__file__))
    model_fp = os.path.join(here, "target_model.pt")
    model, fg = utils.load_model(model_fp, device=device)
    model.eval()

    # grab ~512 examples from val/test
    loader = utils.make_loader('./data/valtest.npz', 'val_x', 'val_y', batch_size=256, shuffle=False)
    x_b, y_b = utils.grab_from_loader(loader, num_batches=2)
    x_b = x_b.to(device)
    y_b = y_b.to(device)
    print("Batch shape:", x_b.shape)

    # pixel epsilons to sweep
    eps_pixels = [8.0/255.0, 12.0/255.0, 16.0/255.0]
    alpha_pixel = 2.0/255.0
    iters = 10

    std = utils.CIFAR10_STD.to(device)
    for eps_pixel in eps_pixels:
        eps_norm = (eps_pixel / std).view(1,3,1,1).to(device)
        alpha_norm = (alpha_pixel / std).view(1,3,1,1).to(device)

        # FGSM
        adv_fgsm = fgsm_attack(model, x_b, y_b, eps_norm, device=device)
        nat_preds = model(x_b).argmax(1).cpu().numpy()
        fgsm_preds = model(adv_fgsm).argmax(1).cpu().numpy()
        nat_acc = (nat_preds == y_b.cpu().numpy()).mean()
        fgsm_acc = (fgsm_preds == y_b.cpu().numpy()).mean()
        print(f"FGSM eps={eps_pixel:.4f} -> nat_acc {nat_acc:.4f}, fgsm_acc {fgsm_acc:.4f}")

        outname = f"advexp_fgsm_eps{int(eps_pixel*255)}.npz"
        np.savez_compressed(outname, adv_x=adv_fgsm.cpu().numpy(), benign_x=x_b.cpu().numpy(), benign_y=y_b.cpu().numpy())
        print("Saved", outname)

        # PGD
        adv_pgd = pgd_attack(model, x_b, y_b, eps_norm, alpha_norm, iters, device=device, rand_start=True)
        pgd_preds = model(adv_pgd).argmax(1).cpu().numpy()
        pgd_acc = (pgd_preds == y_b.cpu().numpy()).mean()
        print(f"PGD eps={eps_pixel:.4f}, iters={iters} -> pgd_acc {pgd_acc:.4f}")

        outname = f"advexp_pgd_eps{int(eps_pixel*255)}_it{iters}.npz"
        np.savez_compressed(outname, adv_x=adv_pgd.cpu().numpy(), benign_x=x_b.cpu().numpy(), benign_y=y_b.cpu().numpy())
        print("Saved", outname)

    print("Done.")
