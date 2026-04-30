```text                                                                
____/\\\\\\\\\________/\\\\\\\\\_________________________________________________        
 __/\\\///////\\\____/\\\///////\\\_______________________________________________       
  _\/\\\_____\/\\\___\/\\\_____\/\\\_______________________________________________      
   _\/\\\\\\\\\\\/____\/\\\\\\\\\\\/________/\\\\\\\\___/\\\____/\\\_____/\\\\\\\\__     
    _\/\\\//////\\\____\/\\\//////\\\______/\\\/////\\\_\//\\\__/\\\____/\\\/////\\\_    
     _\/\\\____\//\\\___\/\\\____\//\\\____/\\\\\\\\\\\___\//\\\/\\\____/\\\\\\\\\\\__   
      _\/\\\_____\//\\\__\/\\\_____\//\\\__\//\\///////_____\//\\\\\____\//\\///////___  
       _\/\\\______\//\\\_\/\\\______\//\\\__\//\\\\\\\\\\____\//\\\______\//\\\\\\\\\\_ 
        _\///________\///__\///________\///____\//////////______\///________\//////////__
```
[](https://opensource.org/licenses/MIT)
## ⇁ TOC

  - [Reve](https://www.google.com/search?q=%23reve)
  - [⇁ TOC](#⇁-toc)
  - [⇁ Who is this for?](#⇁-who-is-this-for)
  - [⇁ Description and features](#⇁-description-and-features)
  - [⇁ Installation](#⇁-installation)
  - [⇁ License](#⇁-license)

## ⇁ Who is this for?

`reve` is designed for researchers, scientists, and students in computational materials science, condensed matter physics, and physical chemistry, particularly those who:

  * Work with atomistic simulation data from molecular dynamics (MD) or Monte Carlo (MC) simulations, especially in formats like XYZ and LAMMPS.
  * Need to perform detailed structural analysis of materials, including disordered systems like glasses and amorphous solids.
  * Investigate various structural properties such as pair distribution functions, bond angular distributions, polyhedricity, and structural units.

## ⇁ Description and Features

`reve` offers a comprehensive suite of tools for the structural analysis of atomic systems.

-----

### **Analysis Capabilities**

`reve` provides various analyzers to quantify structural properties of your systems:

  * **Bond Angular Distribution**: Calculates the distribution of angles between bonded atoms.
  * **Connectivity Analysis**: Determines the connectivity of atoms within the system.
  * **Neutron Structure Factor**: Computes the neutron structure factor, `S(Q)`. A Fast Fourier Transform (FFT) variant is also available.
  * **Pair Distribution Function (PDF)**: Calculates the radial pair distribution function, `g(r)`.
  * **Polyhedricity Analysis**: Quantifies the polyhedricity of local atomic environments.
  * **Structural Units Analysis**: Identifies and characterizes specific structural units within the material.

-----

### **Efficient and Extensible Framework**

  * **I/O**: Includes efficient readers for **XYZ** and **LAMMPS** trajectory files.
  * **Configuration**: Utilizes dataclasses for clear and robust configuration management.
  * **Performance Tracking**: The package can track and save detailed performance metrics.

## ⇁ Installation

### Basic installation

To install `reve` as a package, you can use pip:

```bash
pip install reve
```

Note: the package does not auto upgrade itself, please run the following command to upgrade to the latest version:

```bash
pip install reve --upgrade
```

### Installation from the source code

If you want to install the package from the source code to implement your extensions for example, you can clone the repository:

```bash
git clone git@github.com:jperradin/reve.git
```

Then install the package in development mode:

```bash
cd reve
pip install -e .
```

## ⇁ License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
