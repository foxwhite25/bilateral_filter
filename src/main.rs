use bilateral_filter::SingelChannelImage;
use log::info;
use rayon::prelude::*;

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Trace)
        .init();

    (1..4)
        .into_par_iter()
        .map(|x| 4.0f64.powi(x))
        .for_each(|i| {
            let img = SingelChannelImage::open("lenna.png").unwrap();
            img.gaussian_convolution_v2(i)
                .save(format!("outputs/lenna_conv_{}.png", i))
                .unwrap();
            (-2..=0)
                .into_par_iter()
                .map(|x| 4.0f64.powi(x))
                .for_each(|j| {
                    let img = SingelChannelImage::open("lenna.png").unwrap();
                    img.bilateral_filter(i, j)
                        .save(format!("outputs/lenna_bi_{}_{}.png", i, j))
                        .unwrap();
                    info!("Done {} {}", i, j)
                })
        })
}
