# BME-570-project-code
Matlab code to evaluate performance on a signal-detection task.

## Usage
      [AUC] = performance_evaluation(IS,IN)

## Discription
Conduct an observer study using channelized Hotelling observer.
The signal-present images are given by IS, the signal-absent are images given by
IN. Then compute the AUC, and plot the ROC curves (by simple 
thresholding).

## Inputs 
      IS, IN -- [Number_of_Pixels X Number_of_Images] -- The input images 
                (Each column represents one image)

## Outputs
      AUC -- Wilcoxon area under the ROC curve
      Figures -- ROC curve and delta f_hat_bar image.
      (delta f_hat_bar: the difference between signal-present and signal-absent images)
## Example of output

![output](output.png)

Edited by Zitong Yu @ Nov-16-2021

Based on IQmodelo toolbox Apr-15-2016© the Food and Drug Administration (FDA) and IMAGE QUALITY TOOLBOX Version 0.9b Mar-25-2001© The University of Arizona

Contact: Zitong Yu: yu.zitong@wustl.edu
