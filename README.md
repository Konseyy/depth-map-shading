# Equirect normal maps from depth maps

This is a CLI tool for shading 3D scenes given in equirectangular depth map format.
# Example

```
$ cargo run -r -- -i ./example/inputs/in1.png -o my_output.jpg -k 50
```

This will generate a normal map from the `./example/inputs/in1.png` equirectangular depth map and save it as `./my_output.jpg`.

### Disclaimer

The above command runs the tool in release mode, using all available threads, as the calculation is expensive

# Usage

The usage instructions can be seen by running the following command:

```
$ cargo run -- --help
```

or

```
$ cargo run -- -h
```
