# Image Processing in Rust

This project performs image processing operations, specifically Gaussian convolution and bilateral filtering, on a single channel image. The implementation is in Rust and uses the `image` and `rayon` crates for image manipulation and parallel processing respectively.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You will need Rust and Cargo installed on your machine. Visit the [Rust Programming Language](https://www.rust-lang.org/tools/install) website for installation instructions.

### Installing and Running

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/fox_white25/bilateral_filter.git
    cd bilateral_filter
    ```

2. Build the project:

    ```bash
    cargo build --release
    ```

3. Run the project (replace "lenna.png" in main.rs with the path to your image if you want to):

    ```bash
    cargo run --release
    ```

This will create processed versions of the input image in the `outputs` directory. The naming convention of the output files is `lenna_conv_*.png` for Gaussian convolution and `lenna_bi_*_*.png` for bilateral filtering, where the asterisks represent the standard deviations used.

## Understanding the Code

The `lib.rs` file contains the implementation of the `SingelChannelImage` struct, which represents a single channel image and includes methods for Gaussian convolution and bilateral filtering. The `main.rs` file opens an image, performs the image processing operations, and saves the processed images.

The Gaussian convolution operation applies a Gaussian blur to the image. The amount of blur is controlled by the standard deviation ($\sigma$).

The bilateral filter operation also blurs the image but preserves edges. It uses two standard deviations: a spatial standard deviation ($\sigma_s$) which controls the width of the smoothing region, and a range standard deviation ($\sigma_r$) which controls the amount of smoothing.
