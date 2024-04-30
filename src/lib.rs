use std::{f64::consts::PI, ops::Sub, path::Path};

use anyhow::Ok;
use image::{GenericImageView, Luma};
use indicatif::ParallelProgressIterator;
use rayon::prelude::*;

#[derive(Clone)]
pub struct SingelChannelImage {
    pixels: Vec<f64>,
    width: u32,
    height: u32,
}

impl SingelChannelImage {
    pub fn new(width: u32, height: u32) -> Self {
        SingelChannelImage {
            width,
            height,
            pixels: vec![0.0; width as usize * height as usize],
        }
    }
    pub fn open<T: AsRef<Path>>(path: T) -> anyhow::Result<Self> {
        let original = image::open(path)?;
        let width = original.width();
        let height = original.height();
        let pixels = original
            .pixels()
            .map(|(_, _, color)| {
                0.2126 * color[0] as f64 + 0.7152 * color[1] as f64 + 0.0722 * color[2] as f64
            })
            .collect::<Vec<_>>();
        Ok(Self {
            pixels,
            width,
            height,
        })
    }

    pub fn save<T: AsRef<Path>>(&self, path: T) -> anyhow::Result<()> {
        let mut image = image::GrayImage::new(self.width, self.height);
        image
            .pixels_mut()
            .zip(self.pixels.iter())
            .for_each(|(pixel, lum)| *pixel = Luma::from([*lum as u8]));
        image.save(path)?;
        Ok(())
    }

    pub fn pixel_mut(&mut self, x: u32, y: u32) -> Option<&mut f64> {
        self.pixels.get_mut((y * self.width + x) as usize)
    }

    pub fn guass_convolution(mut self, n_size: f64) -> Self {
        let cache = self.clone();
        let pixel = self.pixels.len();
        let pixel = self
            .pixels()
            .progress_count(pixel as u64)
            .map(|(p_coor, _)| {
                let conv_lum: f64 = cache
                    .pixels()
                    .map(|(q_coor, q_lum)| q_lum * (p_coor - q_coor).guass_kernel(n_size))
                    .sum();
                conv_lum
            })
            .collect::<Vec<_>>();
        self.pixels = pixel;
        self
    }

    pub fn pixels(&self) -> impl ParallelIterator<Item = (Coordinate, f64)> + '_ {
        let w = self.width;
        self.pixels
            .par_iter()
            .enumerate()
            .map(move |(idx, lum)| (coordinate(w, idx), lum.to_owned()))
    }

    pub fn pixels_mut(&mut self) -> impl Iterator<Item = (Coordinate, &mut f64)> {
        let w = self.width;
        self.pixels
            .iter_mut()
            .enumerate()
            .map(move |(idx, lum)| (coordinate(w, idx), lum))
    }

    pub fn gaussian_convolution_v2(&self, n_size: f64) -> Self {
        let mut result = self.clone();
        let kernel = generate_gaussian_kernel(n_size, 6.0);
        let half_kernel_size = (kernel.len() / 2) as i32;
        result.convolve_horizontal(&kernel, half_kernel_size);
        result.convolve_vertical(&kernel, half_kernel_size);

        result
    }

    fn convolve_horizontal(&mut self, kernel: &[f64], half_kernel_size: i32) {
        let width = self.width as i32;
        let pixels = &mut self.pixels;

        pixels.par_chunks_mut(self.width as usize).for_each(|row| {
            let mut row_result = vec![0.0; row.len()];
            for x in 0..width {
                let mut conv_sum = 0.0;
                for i in -half_kernel_size..half_kernel_size {
                    let idx = (x + i).clamp(0, width - 1) as usize;
                    conv_sum += row[idx] * kernel[(i + half_kernel_size) as usize];
                }
                row_result[x as usize] = conv_sum;
            }
            row.copy_from_slice(&row_result);
        });
    }

    fn convolve_vertical(&mut self, kernel: &[f64], half_kernel_size: i32) {
        let width = self.width as i32;
        let height = self.height as i32;
        let len = self.pixels.len();
        let pixels = &mut self.pixels;

        let mut result = vec![0.0; len];
        for y in 0..height {
            for x in 0..width {
                let mut conv_sum = 0.0;
                for i in -half_kernel_size..half_kernel_size {
                    let idx = (y + i).clamp(0, height - 1) as usize;
                    conv_sum += pixels[idx * width as usize + x as usize]
                        * kernel[(i + half_kernel_size) as usize];
                }
                result[(y * width + x) as usize] = conv_sum;
            }
        }
        pixels.copy_from_slice(&result);
    }

    pub fn bilateral_filter(&self, spatial_sigma: f64, intensity_sigma: f64) -> Self {
        let mut result = self.clone();
        let spatial_kernel = generate_gaussian_kernel(spatial_sigma, 6.0);
        let intensity_kernel = generate_gaussian_kernel(intensity_sigma, 20.0);
        let width = self.width as i32;
        let height = self.height as i32;
        let pixels = &self.pixels;
        let half_kernel_size = spatial_kernel.len() as i32 / 2;
        let mut horizontal_filtered_pixels = vec![0.0; self.pixels.len()];
        let mut final_filtered_pixels = vec![0.0; self.pixels.len()];

        // Horizontal filtering
        for y in 0..height {
            for x in 0..width {
                let center_intensity = pixels[(y * width + x) as usize];
                let mut weight_sum = 0.0;
                let mut filtered_pixel = 0.0;

                for i in -half_kernel_size..half_kernel_size {
                    let neighbor_x = (x + i).clamp(0, width - 1);
                    let neighbor_intensity = pixels[(y * width + neighbor_x) as usize];

                    let spatial_weight = spatial_kernel[i as usize + half_kernel_size as usize];
                    let idx = ((center_intensity - neighbor_intensity).abs()
                        * intensity_kernel.len() as f64) as usize;
                    let intensity_weight =
                        intensity_kernel[idx.clamp(0, intensity_kernel.len() - 1)];

                    let weight = spatial_weight * intensity_weight;

                    filtered_pixel += neighbor_intensity * weight;
                    weight_sum += weight;
                }

                horizontal_filtered_pixels[(y * width + x) as usize] = filtered_pixel / weight_sum;
            }
        }

        // Vertical filtering
        for x in 0..width {
            for y in 0..height {
                let center_intensity = horizontal_filtered_pixels[(y * width + x) as usize];
                let mut weight_sum = 0.0;
                let mut filtered_pixel = 0.0;

                for j in -half_kernel_size..half_kernel_size {
                    let neighbor_y = (y + j).clamp(0, height - 1);
                    let neighbor_intensity =
                        horizontal_filtered_pixels[(neighbor_y * width + x) as usize];

                    let spatial_weight =
                        spatial_kernel[(j + spatial_kernel.len() as i32 / 2) as usize];
                    let idx = ((center_intensity - neighbor_intensity).abs()
                        * intensity_kernel.len() as f64) as usize;
                    let intensity_weight =
                        intensity_kernel[idx.clamp(0, intensity_kernel.len() - 1)];

                    let weight = spatial_weight * intensity_weight;

                    filtered_pixel += neighbor_intensity * weight;
                    weight_sum += weight;
                }

                final_filtered_pixels[(y * width + x) as usize] = filtered_pixel / weight_sum;
            }
        }

        // for i in 0..final_filtered_pixels.len() {
        //     let tmp = pixels[i] - final_filtered_pixels[i];
        //     final_filtered_pixels[i] = tmp.clamp(0.0, 255.0)
        // }

        result.pixels = final_filtered_pixels;
        result
    }
}

pub fn coordinate(width: u32, index: usize) -> Coordinate {
    let index = index as u32;
    let x = index % width;
    let y = index / width;
    Coordinate {
        x: x as f64,
        y: y as f64,
    }
}

fn generate_gaussian_kernel(sigma: f64, mult: f64) -> Vec<f64> {
    let size = (mult * sigma).ceil() as usize;
    let mut kernel = Vec::with_capacity(size);
    let factor = 1.0 / (2.0 * PI * sigma * sigma);
    let exp_factor = 1.0 / (2.0 * sigma * sigma);
    let mut sum = 0.0;

    for i in 0..size {
        let x = (i as isize - size as isize / 2) as f64;
        let value = factor * (-x * x * exp_factor).exp();
        sum += value;
        kernel.push(value);
    }

    for value in &mut kernel {
        *value /= sum;
    }

    kernel
}

#[derive(Clone, Copy, Debug)]
pub struct Coordinate {
    x: f64,
    y: f64,
}

impl Sub for Coordinate {
    type Output = Coordinate;

    fn sub(self, rhs: Self) -> Self::Output {
        Coordinate {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl Coordinate {
    pub fn new(x: f64, y: f64) -> Self {
        Coordinate { x, y }
    }

    pub fn euclidian_distance(&self, other: &Self) -> f64 {
        ((self.x - other.y).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }

    pub fn guass_kernel(self, n_size: f64) -> f64 {
        let circle = 1.0 / (2.0 * PI * n_size.powi(2));
        let second = f64::exp(-((self.x.powi(2) + self.y.powi(2)) / (2.0 * n_size.powi(2))));
        circle * second
    }
}
