# crow-tflite-object-detect

Object detection API server using crow webserver.

## Usage

```
$ curl -X POST http://localhost:8888/upload -F "file=@grace_hopper.png;type=image/png"
[{"label":"bow tie","index":458,"probability":0.996078}]
```

## Requirements

* TensorFlow Lite
* OpenCV4

## Installation

```
$ make
```

## License

MIT

## Author

Yasuhiro Matsumoto (a.k.a. mattn)
