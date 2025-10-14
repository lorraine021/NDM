# NDM
A Noise-driven Detection and Mitigation Framework Against Sexual Content in Text-to-Image Generation.



<div align="center">
    <img src="example/cover.png" alt="background" style="width: 90%;"> 
</div>


## Overview
**NDM** is a light-weight noise-driven framework, which could detect and mitigate both explicit and implicit sexual intention in T2I generation. 
We uncover two key insights into noises for safe text-to-image generation: **the separability of early-stage predicted noises** (allowing for efficient detection) and **the significant impact of initial noises** on sexual content generation (leading to a more effective noise-enhanced adaptive negative guidance for mitigation).

<div align="center">
    <img src="example/framework.png" alt="background" style="width: 90%;"> 
</div>

## 🚀 News
- **2025.10**: 🌟 We further provide theoretical justification for the separability of latent representations and extend our method to a broader range of T2I models. The new version will be released soon!
- **2025.07**: 🌟 Our paper "NDM: A Noise-driven Detection and Mitigation Framework against Implicit Sexual Intentions in Text-to-Image Generation" has been accepted by ACMMM2025!



## 🛠️ Environment

### Requirements
- **Python**: 3.10+


### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/lorraine021/NDM.git
   cd NDM
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Generation
    ```bash
    python run_ndm.py
    ```

## Contributing
We welcome contributions! Please submit issues or pull requests for bug fixes, features, or documentation enhancements.

