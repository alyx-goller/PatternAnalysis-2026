import os, sys
import torch
import numpy as np
from tqdm import tqdm #to help visualise loading
import time
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
 
from dataset import get_dataloaders, NUM_CLASSES, PROSTATE_CLASS
from modules import UNet2D, DiceLoss, dice_counts
 
#define HIPMRI_ROOT in job file when running on remote HPC
ROOT = os.environ.get("HIPMRI_ROOT", "/Users/alyxgoller/data/HipMRI_data")  #changes per machine
WEIGHTS = "best_model.pth"
NUM_OF_WORKERS = 0   # raise to around 4 on the HPC


#set up gpu access
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("Warning: CUDA not found. Using CPU")

@torch.no_grad()
def general_predict(batch_size = 16):
    """
    Main test function for segmenting images and evaluating performance and cost
    
    """
    
    
 
    # -- fetch test images & their labels --
    _, _, test_loader = get_dataloaders(ROOT, batch_size=batch_size, num_workers=NUM_OF_WORKERS)
 
    #prepare model and load weights
    model = UNet2D(in_channels=1, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    model.eval() #turn off training mode/flag
 
    # -- predict and evaluate --
    best_dice = 0
    worst_dice = np.inf
    best_batch = ()
    worst_batch = ()
    total = 0.0
    intersection = torch.zeros(NUM_CLASSES, device=device)
    total = torch.zeros(NUM_CLASSES, device=device)   
    for image, label in tqdm(test_loader, desc="test", leave=False, disable=not sys.stdout.isatty()):
        image, label = image.to(device), label.to(device)
        
        #predict bacth
        predicted_mask = model(image)   # raw scores (logits) - argmax later turns them into a mask
        
        # -- Evaluating --
 
        #store overall dice data for averaging - for every class
        i, t = dice_counts(predicted_mask, label)
        intersection += i
        total += t
 
        #determine if best or worst dice score so far & store the best and worst batches for final visualisation
        #(compared on the prostate class only, so each batch gives one number)
        cur_batch_dice = ((2 * i[PROSTATE_CLASS] + 1e-5) / (u[PROSTATE_CLASS] + 1e-5)).item()
        if label[:, PROSTATE_CLASS].sum() == 0:
            continue   # a batch with no prostate gives a meaningless score, so it can't be best or worst
        if cur_batch_dice > best_dice:
            best_dice = cur_batch_dice
            best_batch = (image, label, predicted_mask, i, u)
        if cur_batch_dice < worst_dice:
            worst_dice = cur_batch_dice
            worst_batch = (image, label, predicted_mask, i, u)
 
    # -- Reporting --
    ave_dice = (2 * intersection + 1e-5) / (total + 1e-5)
    
    print("Dice figures:")
    print("Average (per class): ", ave_dice, " Best batch prostate Dice: ", best_dice, " Worst batch prostate Dice: ", worst_dice)
    print("Prostate average: ", ave_dice[PROSTATE_CLASS].item())
 
    if len(best_batch) == 0:
        print("No test batch contained prostate, nothing to visualise")
        return
 
    #visualise slices from best and worst batches with num of slices specified and if prostate = True only show slices if prostate visible
    visualise(best_batch, worst_batch, num_of_slices = 3, prostate = True)
 






# -- Visualisation Helpers --
 
def dice_per_class(pred, true):
    """Calculate Dice of every class for ONE slice.
        
       Args:
           pred = predicted segmentation
           true = true mask - segmented label of image
        Returns:
            List of Dice scores for each class in the slice - nan for classes not present in slice
        """
    scores = []
    for c in range(NUM_CLASSES):
        both = ((pred == c) & (true == c)).sum()
        total = (pred == c).sum() + (true == c).sum()
        scores.append(float("nan") if total == 0 else float(2 * both / total))
    return scores
 
 
def pick_slices(batch, title, num_of_slices, prostate):
    """Obtain key information for plotting and add to easily accessible list:
        Num_of_slices slices from one stored batch -> list of (title, image, true mask, predicted mask)

        Args:
            batch = store information (image, label, predicted mask of image, intersection, total area)
            title = name of image - either BEST or WORST in case of predict plotting
            num_of_slices = the number of slices that are to be plotted/visualised
            prostate (Bool) = True if only selecting slices with a prostate present in label, else False (ignore missing prostate)

        Return:
            List of (title, image, true mask, predicted mask)
    """
    image, label, predicted_mask, i, u = batch
    image = image.cpu().numpy()
    true = label.argmax(dim=1).cpu().numpy()             # one-hot channels -> one class number per pixel
    pred = predicted_mask.argmax(dim=1).cpu().numpy()    # highest scoring class per pixel
    batch_dice = ((2 * i[PROSTATE_CLASS] + 1e-5) / (u[PROSTATE_CLASS] + 1e-5)).item()
 
    rows = []
    for s in range(len(image)):
        # skip slices where the prostate isn't in the label
        if prostate and not (true[s] == PROSTATE_CLASS).any():
            continue   
        
        rows.append((f"{title} batch (prostate Dice {batch_dice:.3f})\nslice {s}", image[s, 0], true[s], pred[s]))
        if len(rows) == num_of_slices:
            break
    return rows


#Visualising Functions written primarily by AI

def draw_rows(rows):
    """One figure, one row per slice: input | ground truth | prediction | prostate errors | Dice per class"""
    class_colours = ["black", "tab:blue", "tab:green", "tab:orange", "tab:red", "tab:purple"]
    classes = ListedColormap(class_colours)
    error_colours = ListedColormap(["black", "green", "red", "blue"])
    fig = Figure(figsize=(14, 5.5 * len(rows)), dpi=100)   # tall figure, scrolled in the window
 
    for r, (title, image, true, pred) in enumerate(rows):
        ax = fig.add_subplot(len(rows), 5, 5 * r + 1)
        ax.imshow(image, cmap="gray")
        ax.set_title("Input")
 
        ax = fig.add_subplot(len(rows), 5, 5 * r + 2)
        ax.imshow(true, cmap=classes, vmin=0, vmax=NUM_CLASSES - 1, interpolation="nearest")
        ax.set_title("Ground truth")
 
        ax = fig.add_subplot(len(rows), 5, 5 * r + 3)
        ax.imshow(pred, cmap=classes, vmin=0, vmax=NUM_CLASSES - 1, interpolation="nearest")
        ax.set_title("Prediction")
 
        #prostate errors: 1 = correct (green), 2 = false positive (red), 3 = missed (blue), 0 = nothing (hidden)
        p, t = pred == PROSTATE_CLASS, true == PROSTATE_CLASS
        errors = 1 * (p & t) + 2 * (p & ~t) + 3 * (~p & t)
        ax = fig.add_subplot(len(rows), 5, 5 * r + 4)
        ax.imshow(image, cmap="gray")
        ax.imshow(np.ma.masked_equal(errors, 0), cmap=error_colours, vmin=0, vmax=3, alpha=0.8, interpolation="nearest")
        ax.set_title("Prostate: green correct,\nred false +, blue missed")
 
        ax = fig.add_subplot(len(rows), 5, 5 * r + 5)
        text = title + "\n\nDice per class\n"
        for c, d in enumerate(dice_per_class(pred, true)):
            text += f"class {c}: {d:.3f}" + ("  <- prostate" if c == PROSTATE_CLASS else "") + "\n"
        ax.text(0, 1, text, va="top", family="monospace", fontsize=11)
 
    for ax in fig.axes:
        ax.axis("off")
    fig.legend(handles=[Patch(color=col, label=f"class {c}") for c, col in enumerate(class_colours)],
               loc="upper center", ncol=NUM_CLASSES)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, wspace=0.05, hspace=0.2)
    return fig
 
 
def show_scrollable(fig):
    """Put the figure in a window with a scrollbar (mouse wheel works too)"""
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
 
    window = tk.Tk()
    window.geometry("1300x800")
    canvas = tk.Canvas(window)
    scrollbar = tk.Scrollbar(window, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
 
    figure_canvas = FigureCanvasTkAgg(fig, master=canvas)
    widget = figure_canvas.get_tk_widget()
    canvas.create_window((0, 0), window=widget, anchor="nw")
    figure_canvas.draw()
    canvas.configure(scrollregion=(0, 0, widget.winfo_reqwidth(), widget.winfo_reqheight()))
 
    canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-1 if e.delta > 0 else 1, "units"))  # Mac / Windows
    canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))                           # Linux up
    canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))                            # Linux down
    window.mainloop()   # stays open until the window is closed
 
 
def visualise(best_batch, worst_batch, num_of_slices = 3, prostate = True):
    """Show num_of_slices slices from the best batch, then from the worst batch, in one scrollable window.
    prostate = True -> only slices where the prostate is in the label"""
    rows = pick_slices(best_batch, "BEST", num_of_slices, prostate) + pick_slices(worst_batch, "WORST", num_of_slices, prostate)
    fig = draw_rows(rows)
    try:
        show_scrollable(fig)
    except Exception:   # no window possible (e.g. terminal-only HPC) -> save a picture instead
        fig.savefig("best_and_worst.png", bbox_inches="tight")
        print("No window available, saved best_and_worst.png instead")

    
    

    

      
    



    
    

if __name__ == "__main__":
    general_predict(batch_size=16)