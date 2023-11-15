# Artificial Artwork

## Quick-start

Run a demo NST, on sample `Content` and `Style` Images:

```shell
mkdir art
export NST_HOST_MOUNT="$PWD/art"

# Run containerized NST, and exit container upon finish
docker-compose up
```

Check out your **Generated Image**!  
Artificial Artwork: **art/canoe_water_w300-h225.jpg+blue-red_w300-h225.jpg-100.png**  

```shell
xdg-open art/canoe_water_w300-h225.jpg+blue-red_w300-h225.jpg-100.png
```

## Usage

Run the `nst` CLI with the `--help` option to see the available options.

```shell
docker run boromir674/neural-style-transfer:1.0.2 --help
```

## Development


### Installation from `pypi`

```shell
pip install artificial-artwork
```

Only python3.8 wheel is included atm.

### Installation from `source`:

```shell
git clone https://github.com/boromir674/neural-style-transfer.git
pip install ./neural-style-transfer
```

The Neural Style Transfer - CLI heavely depends on Tensorflow (tf) and therefore it is crucial that tf is installed correctly in your Python environment.
