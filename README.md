# OAK-D SR Test Lab

Detta projekt startar arbetet med Luxonis OAK-D SR. Forsta malet ar att
verifiera USB-kontakt med kameran och fa upp en stabil bild i webblasaren.

OAK-D SR ar en kortdistans-stereokamera med tva OV9782 global-shutter-sensorer
pa upp till `1280x800`. Den har kort stereobaslinje, cirka `20 mm`, och ar
tankt for kortare arbetsavstand.

## Installation

```bash
cd ~/projects/oak_d_sr
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Kontrollera kameran

Med kameran ansluten via USB:

```bash
python scripts/oakdsr_connect_check.py
```

Om flera OAK-enheter ar anslutna kan du ange MXID:

```bash
python scripts/oakdsr_connect_check.py --device <mxid>
```

Om DepthAI ser USB-enheten men skriver `Insufficient permissions` eller
`Make sure udev rules are set`, installera udev-regeln pa den Raspberry Pi dar
kameran sitter:

```bash
scripts/oakdsr_install_udev_rules.sh
```

Koppla sedan ur och in kameran igen och kor connect-check en gang till.

## Livestream

Starta en enkel webbaserad bildstrom:

```bash
python scripts/oakdsr_livestream.py --web-host 0.0.0.0
```

Oppna fran Macen:

```text
http://johannesmelin.local:8092
```

Om namnet inte hittas kan du anvanda Pi:ns IP-adress:

```text
http://<pi-ip>:8092
```

Scriptet valjer automatiskt forsta anslutna kamera-socket. Om du vill tvinga
en viss sensor:

```bash
python scripts/oakdsr_livestream.py --web-host 0.0.0.0 --socket cam_b
python scripts/oakdsr_livestream.py --web-host 0.0.0.0 --socket cam_c
```

Vanliga parametrar:

- `--width`, `--height`: bildstorlek. Standard ar `640x400`.
- `--fps`: kamerans bildfrekvens. Standard ar `15`.
- `--socket`: `auto`, `cam_a`, `cam_b` eller `cam_c`.
- `--camera-rotation`: `0` eller `180`.
- `--viewer-scale`: skalar bilden som skickas till browsern.
- `--jpeg-quality`: JPEG-kvalitet for browserbilden.
- `--web-host`: satt till `0.0.0.0` for att na sidan fran annan dator.
- `--web-port`: standard `8092`.

For hogre detaljkontroll:

```bash
python scripts/oakdsr_livestream.py --web-host 0.0.0.0 \
  --width 1280 --height 800 --fps 10 --viewer-scale 0.75
```

## YOLO-segmentering med stereo-depth

OAK-D SR har ingen separat `CAM_A`-RGB-kamera. Scriptet
`scripts/oakdsr_yolo_seg_localizer.py` anvander darfor vanster stereokamera
`CAM_B` som kamerabild, `CAM_C` som hoger stereokamera och alignar depth till
`CAM_B`.

Som standard laddas den senaste segmenteringsmodellen fran `oak_camera`:

```text
/home/johannes/projects/oak_camera/training_runs/knopp_yolo_seg_640_e30/weights/best.pt
```

Starta med full sensorupplosning, roterad bild och en webbild per sekund:

```bash
/home/johannes/projects/oak_camera/.venv/bin/python \
  scripts/oakdsr_yolo_seg_localizer.py --web-host 0.0.0.0 \
  --width 1280 --height 800 \
  --stereo-width 1280 --stereo-height 800 \
  --fps 2 --camera-rotation 180 \
  --viewer-interval-ms 1000 --viewer-scale 0.75
```

Forsta testet kors med `oak_camera/.venv` eftersom den redan har fungerande
CPU-versioner av `ultralytics`, `torch` och `torchvision`. Undvik att installera
senaste `ultralytics` rakt av i Pi-miljon, eftersom den kan dra in onodigt stora
CUDA-relaterade PyTorch-paket.

Oppna:

```text
http://johannesmelin.local:8094
```

Programmet gor foljande:

1. Laser bild fran `CAM_B`.
2. Beraknar stereo-depth fran `CAM_B` och `CAM_C`.
3. Alignar depth-bilden till `CAM_B`.
4. Roterar bade kamerabild och depth `180` grader om `--camera-rotation 180`
   anvands.
5. Kor YOLO-segmenteringsmodellen pa kamerabilden.
6. Anvander segmenteringsmasken som depth-ROI.
7. Beraknar `z` som median av de framsta giltiga depth-pixlarna i masken.
8. Raknar om maskens depth-pixel till `x,y,z` med kamerans intrinsics.

Viktiga parametrar:

- `--model`: modellfil. Standard ar senaste segmenteringsmodellen fran
  `oak_camera`.
- `--classes`: klassfilter. Standard `knopp`.
- `--lower-mm`, `--upper-mm`: godkant depth-intervall. Standard
  `150-1200 mm`.
- `--mask-scale`: krymper masken runt objektets centrum innan depth mats.
  Standard `0.8`.
- `--min-depth-pixels`: minsta antal giltiga depth-pixlar i masken. Standard
  `1`.
- `--depth-percentile`: valjer narmre depth-varden i masken. Standard `20`.
- `--depth-band-mm`: bredd runt vald percentile som anvands for median. Standard
  `30`.
- `--smooth-window`: medianutjamning av `x,y,z`. Standard `1`.
- `--viewer-interval-ms`: hur ofta browserbilden uppdateras. Standard
  `1000`.
