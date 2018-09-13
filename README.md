# Air-Written-Digit-Recognition
Air wriiten digit prediction

how it works?
https://www.youtube.com/watch?v=tEhszmidQEQ

find the attached pdf
Air written digit classification:


Procedure in a nutshell:
1) Load the classifier using joblib (pickle file)
2) Read the input image
3)Convert to grayscale and apply Gaussian filtering
4)Threshold the image
5) Find contours in the image
6) Get rectangles contains each contour
 6.1)For each rectangular region, calculate HOG features and predict
             the digit using LINEAR SVM 
6.1.1:    DRAW
6.1.2:    RESIZE
6.1.3:    HOG



 PICKLE FILE:
 object in python can be pickled so that it can be saved on disk.  Pickling is a way to convert a python object (list, dict, etc.) into a character stream. The idea is that this character stream contains all the information necessary to reconstruct the object in another python script.



  
 USING HISTOGRAM OF ORIENTED GRADIENT:

(optional) global image normalisation
computing the gradient image in x and y
computing gradient histograms
normalising across blocks
flattening into a feature vector


LINEAR SVM:

Support vector machine constructs a hyperplane or set of hyperplanes in a high- or infinite-dimensional space, which can be used for classification, regression, or other tasks like outliers detection. Intuitively, a good separation is achieved by the hyperplane that has the largest distance to the nearest training-data point of any class (so-called functional margin), since in general the larger the margin the lower the generalization error of the classifier.
Experimental results show that SVMs achieve significantly higher search accuracy than traditional query refinement schemes after just three to four rounds of relevance feedback. This is also true of image segmentation systems, including those using a modified version SVM that uses the privileged approach as suggested by Vapnik.


HOW MY PROJECT WORKS
I have attached a video of how exactly my project works.Hope it helps. 



