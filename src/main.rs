use clap::Parser;
use image::codecs::jpeg::JpegEncoder;
use image::{self, ImageEncoder, Rgb};
use kd_tree::KdTree;
use rayon::prelude::*;
use std::path::PathBuf;
use std::time::{Duration, Instant};

mod utils;

use utils::{
    process_point, spherical_to_cartesian, text_coords_to_spherical, CartesianCoordinate,
    NearPointAlgorithm, OutputType,
};

use crate::utils::ProcessedPoint;

// CLI Parameters
// -----------
#[derive(Parser, Debug)]
#[command( version, about="This is a CLI tool for generating normal maps, given an equirectangular format depth map", long_about = None )]
struct Args {
    // Input depth map file path
    #[arg(name = "Input file path (depth map)", short = 'i', long = "input")]
    input_path: PathBuf,
    // Output normal map file path
    #[arg(name = "Output file path (normal map)", short = 'o', long = "output")]
    output_path: PathBuf,
    // How many nearest neighbors to consider
    #[arg(
        name = "Num of nearest neighbors to consider",
        short = 'k',
        long = "k-nearest",
        default_value_t = 80usize
    )]
    k_nearest: usize,
    // Whether to output normals or absolute normals
    #[arg(
        name = "Output color",
        short = 'c',
        long = "color",
        default_value = "shaded"
    )]
    output_color: OutputType,
    #[arg(
        name = "Nearest point algorithm",
        short = 'a',
        long = "nearest-algorithm",
        default_value = "within_radius"
    )]
    algorithm: NearPointAlgorithm,
}

fn main() {
    let args = Args::parse();
    if args.k_nearest <= 0 {
        panic!("Number of nearest neighbors must be greater than 0");
    }
    // check if output directory exists?
    let time_total_calc = Instant::now();
    let time_start = Instant::now();
    let img = image::open(args.input_path)
        .expect("File not found!")
        .into_luma16();

    let (img_width, img_height) = img.dimensions();

    let mut new_img = image::ImageBuffer::<image::Rgb<u8>, _>::new(img_width, img_height);

    let num_points = img_width * img_height;

    let mut coordinates = Vec::<CartesianCoordinate>::with_capacity((num_points) as usize);

    for (x, y, pixel) in img.enumerate_pixels() {
        let depth = pixel.0[0] as f32;

        let spherical = text_coords_to_spherical(x, y, (img_width, img_height), depth);
        let cartesian = spherical_to_cartesian(&spherical);

        coordinates.push(cartesian);
    }

    let tree = KdTree::par_build_by_ordered_float(coordinates.clone());

    let elapsed_tree_constr = time_start.elapsed();
    println!("KD tree construction took {:.2?}", elapsed_tree_constr);

    println!(
        "Starting normal calculation using {:} nearest neighbors",
        args.k_nearest
    );

    let results = coordinates
        .par_iter()
        .map(|point| {
            process_point(
                &point,
                args.k_nearest,
                &tree,
                args.algorithm,
                args.output_color,
            )
        })
        .collect::<Vec<ProcessedPoint>>();

    let mut elapsed_tree_search = Duration::from_secs(0);
    let mut elapsed_plane_fit = Duration::from_secs(0);

    let writer = std::fs::File::create(&args.output_path).expect("Failed to create file");
    let encoder = JpegEncoder::new_with_quality(writer, 100);

    let elapsed_calc = time_total_calc.elapsed();

    for result_px in &results {
        new_img.put_pixel(
            result_px.text_coords.x,
            result_px.text_coords.y,
            Rgb::<u8>(result_px.rgb),
        );
        elapsed_tree_search += result_px.tree_search_elapsed;
        elapsed_plane_fit += result_px.plane_fit_elapsed;
    }

    encoder
        .write_image(&new_img, img_width, img_height, image::ColorType::Rgb8)
        .expect("Error writing image");

    elapsed_tree_search /= num_points;
    elapsed_plane_fit /= num_points;

    println!(
        "Total calculation took {:.2?} with {:} nearest neighbors",
        elapsed_calc, args.k_nearest
    );
    println!(
        "Average time per pixel for tree search: {:.2?}",
        elapsed_tree_search
    );
    println!(
        "Average time per pixel for plane fit: {:.2?}",
        elapsed_plane_fit
    );
    println!(
        "File has been saved to {}",
        args.output_path.to_str().unwrap()
    );
}
