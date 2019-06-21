### Implementation of O2P2 

Data directory structure:

Initial Final: {'initial_final':{'train': , 'val': }}

#### Choices to try
* In the paper, they mention perception module takes in segmented images. 
    * Input: (img, mask) or (img * mask)
        * Currently trying (img, mask). Try (img * mask) as well. 
* Render module for mask.
    * Putting Sigmoid on top of final convolution.
        * Currently not using sigmoid, mask outputs are logits.
* Physics Interaction Module
    * Combining the two object vectors before passing them through the interaction engine.
        * Currently concatenating them.