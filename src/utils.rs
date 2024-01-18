use clap::ValueEnum;
use core::panic;
use kd_tree::KdPoint;
use nalgebra::{Const, Dyn, Matrix, SVector, VecStorage, SVD};
use std::{
    f32::consts::PI,
    ops::{Add, Mul},
    time::{Duration, Instant},
};

use kd_tree::KdTree;

pub type Vec3 = SVector<f32, 3>;

#[derive(Copy, Clone)]
pub struct TextCoords {
    pub x: u32,
    pub y: u32,
}

pub struct SphericalCoordinate {
    pub theta: f32,
    pub phi: f32,
    pub r: f32,
    pub from_text: TextCoords,
}

#[derive(Copy, Clone)]
pub struct CartesianCoordinate {
    pub vec_coord: Vec3,
    pub from_text: TextCoords,
}

impl KdPoint for CartesianCoordinate {
    type Scalar = f32;
    type Dim = typenum::U3;
    fn at(&self, k: usize) -> f32 {
        match k {
            0 => self.vec_coord.x,
            1 => self.vec_coord.y,
            2 => self.vec_coord.z,
            _ => panic!("Invalid dimension"),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum OutputType {
    Shaded,
    Normals,
    NormalsFull,
}
impl ValueEnum for OutputType {
    fn from_str(input: &str, _ignore_case: bool) -> Result<Self, String> {
        match input {
            "shaded" => Ok(OutputType::Shaded),
            "normals" => Ok(OutputType::Normals),
            _ => Err(format!("Invalid output color: {}", input)),
        }
    }
    fn to_possible_value(&self) -> Option<clap::builder::PossibleValue> {
        match self {
            OutputType::Shaded => Some(clap::builder::PossibleValue::new("shaded")),
            OutputType::Normals => Some(clap::builder::PossibleValue::new("normals")),
            OutputType::NormalsFull => Some(clap::builder::PossibleValue::new("normals_full")),
        }
    }
    fn value_variants<'a>() -> &'a [Self] {
        &[
            OutputType::Shaded,
            OutputType::Normals,
            OutputType::NormalsFull,
        ]
    }
}
#[derive(Copy, Clone, Debug)]
pub enum NearPointAlgorithm {
    KNearest,
    WithinRadius,
}
impl ValueEnum for NearPointAlgorithm {
    fn from_str(input: &str, _ignore_case: bool) -> Result<Self, String> {
        match input {
            "k_nearest" => Ok(NearPointAlgorithm::KNearest),
            "within_radius" => Ok(NearPointAlgorithm::WithinRadius),
            _ => Err(format!("Invalid algorithm: {}", input)),
        }
    }
    fn to_possible_value(&self) -> Option<clap::builder::PossibleValue> {
        match self {
            NearPointAlgorithm::KNearest => Some(clap::builder::PossibleValue::new("k_nearest")),
            NearPointAlgorithm::WithinRadius => {
                Some(clap::builder::PossibleValue::new("within_radius"))
            }
        }
    }
    fn value_variants<'a>() -> &'a [Self] {
        &[
            NearPointAlgorithm::KNearest,
            NearPointAlgorithm::WithinRadius,
        ]
    }
}

pub struct ProcessedPoint {
    pub text_coords: TextCoords,
    pub rgb: [u8; 3],
    pub tree_search_elapsed: Duration,
    pub plane_fit_elapsed: Duration,
}

pub fn text_coords_to_spherical(
    x: u32,
    y: u32,
    dimensions: (u32, u32),
    depth: f32,
) -> SphericalCoordinate {
    // Get percentages
    let perc_x = x as f32 / dimensions.0 as f32;
    let perc_y = y as f32 / dimensions.1 as f32;
    // Convert to spherical coordinates
    return SphericalCoordinate {
        theta: perc_x * 2. * PI,
        phi: perc_y * PI,
        r: depth,
        from_text: TextCoords { x, y },
    };
}

pub fn spherical_to_cartesian(sph: &SphericalCoordinate) -> CartesianCoordinate {
    let vec = Vec3::new(
        sph.r * sph.phi.sin() * sph.theta.cos(),
        sph.r * sph.phi.sin() * sph.theta.sin(),
        sph.r * sph.phi.cos(),
    );
    return CartesianCoordinate {
        vec_coord: vec,
        from_text: sph.from_text,
    };
}

pub fn process_point(
    point: &CartesianCoordinate,
    k: usize,
    tree: &KdTree<CartesianCoordinate>,
    algorithm: NearPointAlgorithm,
    desired_output: OutputType,
) -> ProcessedPoint {
    let tree_search_start = Instant::now();

    let k_nearest = match algorithm {
        NearPointAlgorithm::KNearest => tree
            .nearests(point, k)
            .iter()
            .map(|x| x.item)
            .collect::<Vec<&CartesianCoordinate>>(),
        NearPointAlgorithm::WithinRadius => {
            let mut radius = k as f32;
            let mut points = tree.within_radius(point, radius);
            while points.len() < 10 {
                radius *= 2.;
                points = tree.within_radius(point, radius);
            }
            points
        }
    };

    let tree_search_elapsed = tree_search_start.elapsed();
    let num_points_sampled = k_nearest.len();

    let plane_fit_start = Instant::now();
    let mut sum = Vec3::new(0., 0., 0.);

    for near_point in &k_nearest {
        sum = sum + near_point.vec_coord;
    }

    // Calculate normal of the plane using SVD
    let centroid = sum / num_points_sampled as f32;

    // store the nearest points in a 3 x N matrix where N = number of points sampled
    // and each column is a point
    let matrix = Matrix::<f32, Const<3>, Dyn, VecStorage<f32, Const<3>, Dyn>>::from_fn(
        num_points_sampled,
        |row, col| {
            if col >= k_nearest.len() {
                panic!("Invalid column");
            }
            let point = k_nearest[col];
            match row {
                0 => point.vec_coord.x - centroid.x,
                1 => point.vec_coord.y - centroid.y,
                2 => point.vec_coord.z - centroid.z,
                _ => panic!("Invalid row"),
            }
        },
    );

    let svd = SVD::new(matrix, true, false);

    let plane_fit_elapsed = plane_fit_start.elapsed();

    let left_singular = svd.u.unwrap();

    let normal_col = left_singular.column(2);

    let mut normal_full_vec = normal_col.normalize();

    if normal_full_vec.dot(&point.vec_coord) < 0. {
        // If normal is oriented away from the camera, flip it so it's facing the camera
        normal_full_vec *= -1.;
    }

    const LIGHT_POS: Vec3 = Vec3::new(0., -2500., -100.);

    let light_dir = point.vec_coord - LIGHT_POS;

    let dist_to_light = (light_dir.magnitude() / (256. * 25.)).min(1.);

    let light_level = normal_full_vec.dot(&light_dir.normalize()).max(0.) * (1. - dist_to_light);

    let shaded_rgb = [
        (light_level * 255.) as u8,
        (light_level * 255.) as u8,
        (light_level * 255.) as u8,
    ];

    let normal_abs_vec = normal_col.abs().normalize();

    let normal_abs_rgb = [
        (normal_abs_vec.x * 255.) as u8,
        (normal_abs_vec.y * 255.) as u8,
        (normal_abs_vec.z * 255.) as u8,
    ];

    let normal_col = normal_full_vec.mul(-0.5).add(Vec3::new(0.5, 0.5, 0.5));

    let normal_full_rgb = [
        (normal_col.x * 255.) as u8,
        (normal_col.y * 255.) as u8,
        (normal_col.z * 255.) as u8,
    ];

    let text_coords = point.from_text;
    return ProcessedPoint {
        text_coords,
        rgb: match desired_output {
            OutputType::Shaded => shaded_rgb,
            OutputType::Normals => normal_abs_rgb,
            OutputType::NormalsFull => normal_full_rgb,
        },
        tree_search_elapsed,
        plane_fit_elapsed,
    };
}
