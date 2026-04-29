# Overcoming Local Snapping: Magnetic-Edge Loss & ASPP

## The Limitation: The Aperture Problem & Inward Pulling
During testing, we noticed the Puller model suffered from greedy local snapping. When faced with parallel edges (e.g., a window frame next to the glass) or textured regions, the model would pull the boundary inward or produce wobbly lines. It lacked the global context to choose the *correct* edge and had no mathematical reason to avoid placing boundaries in the middle of flat color zones.

## The Solution
To fix this, we introduced two major spatial upgrades to Iter 7 (In kaggle codes folder):

### 1. Magnetic Gradient Penalty (Sobel-Weighted Loss)
* **Intuition:** We wanted to tell the model, "You are never allowed to put a boundary on a flat color patch." The actual physical edges of the image should act as a magnetic track.
* **Implementation:** We computed the magnitude gradient of the original RGB image using Sobel filters (`img_grad`). We then added a penalty term to the Puller's loss: `pred_boundary * (1.0 - img_grad)`. 
* **Effect:** If the model predicts a boundary where the image has zero color gradient (a flat region), it receives a massive penalty. It is forced to "snap" to high-contrast pixels.

### 2. ASPP-Lite (Atrous Spatial Pyramid Pooling)
* **Intuition:** To stop the model from blindly snapping to the *closest* edge, it needs to see the bigger picture.
* **Implementation:** We injected an ASPP block with dilation rates of 6 and 12 into the Puller's stem. This expanded the receptive field, allowing the network to see parallel lines simultaneously and use global object context to pick the right one.

## Observations
The combination worked exceptionally well. The wobbling vanished, and the boundary successfully snapped to the correct outer wooden frame instead of dipping into the glass. 

Because the magnetic penalty was strict, the model learned to take very small, "safe" steps across flat regions to avoid getting penalized. As a result, achieving the perfect global snap required around 30 iterations. Tuning the magnetic weight (e.g., setting it to 2.0) provided the perfect balance between keeping the edges crisp and allowing the Puller to cross flat zones to find the true object boundary.