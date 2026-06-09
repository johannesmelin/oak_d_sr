# OAK-D SR Test Lab

Detta projekt startar arbetet med Luxonis OAK-D SR. Forsta malet ar att
verifiera USB-kontakt med kameran och fa upp en stabil bild i webblasaren.

OAK-D SR ar en kortdistans-stereokamera med tva OV9782 global-shutter-sensorer
pa upp till `1280x800`. Den har kort stereobaslinje, cirka `20 mm`, och ar
tankt for kortare arbetsavstand.

Aktuell version ar byggd specifikt for OAK-D SR och gor:

- YOLO-segmentering av knoppar i `CAM_B`-bilden.
- stereo-depth med `CAM_B` och `CAM_C`, depth alignad till `CAM_B`.
- HSV-filtrerat depth-urval inne i segmenteringsmasken.
- position i bade kamerakoordinater, `cam x,y,z`, och kalibrerade
  arbetskoordinater, `grid x,y,z`.
- webbaserad kontrollbild med numrerade knoppar och en samlad koordinatlista.

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

Som standard laddas en tidigare segmenteringsmodell fran `oak_camera`:

```text
/home/johannes/projects/oak_camera/training_runs/knopp_yolo_seg_640_e30/weights/best.pt
```

Den aktuella OAK-D SR-versionen ar tranad pa bilder fran OAK-D SR i
`1280x800` och anvands med modellen:

```text
training_runs/knopp_oakdsr_yolo_seg_960_e50/weights/best.pt
```

Starta med full sensorupplosning, roterad bild och en webbild per sekund:

```bash
/home/johannes/projects/oak_camera/.venv/bin/python \
  scripts/oakdsr_yolo_seg_localizer.py --web-host 0.0.0.0 \
  --model training_runs/knopp_oakdsr_yolo_seg_960_e50/weights/best.pt \
  --width 1280 --height 800 \
  --stereo-width 1280 --stereo-height 800 \
  --fps 2 --camera-rotation 180 \
  --viewer-interval-ms 1000 --viewer-scale 1.0 \
  --classes knopp --imgsz 960 \
  --depth-mask hsv \
  --hsv-low 62,35,35 --hsv-high 158,255,255 \
  --hsv-open-kernel 0 --hsv-close-kernel 2 \
  --show-boxes
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
8. Raknar om maskens depth-pixel till `cam x,y,z` med kamerans intrinsics.
9. Raknar om `cam` till kalibrerade `grid`-koordinater med en rigid
   transformation.

Kalibreringstransformen ar:

```text
grid = R @ cam + t

R =
[[ 0.999739296, -0.017369757,  0.014819986],
 [-0.022243652, -0.594407549,  0.803856260],
 [-0.005153677, -0.803976342, -0.594638951]]

t = [10.975328, -4.674079, 317.735563]
```

Transformen ar anpassad till 45 uppmatta kalibreringspunkter for den aktuella
OAK-D SR-uppstallningen. Anpassningsfelet blev:

```text
3D RMS-fel: cirka 6.1 mm
Medelfel:  cirka 5.5 mm
Maxfel:    cirka 12.6 mm
```

I webblasaren markeras varje knopp med ett nummer, `1`, `2`, `3` osv. I
vansterkanten visas en samlad lista med bade `cam x,y,z` och `grid x,y,z` for
respektive nummer. Detta gor att texten inte overlappar nar flera knoppar star
nara varandra och gor det enklare att samla ny kalibreringsdata.

Om bara kamerakoordinater ska visas kan grid-transformen stangas av med:

```bash
--no-grid-transform
```

Depth-ROI kan ocksa kombineras med HSV-filter. Da anvands bara pixlar som bade
ligger i YOLO-segmenteringsmasken och matchar objektets HSV-farg. Syftet ar att
minska risken att depth tas fran bakgrund eller skymmande detaljer nar objektet
bara delvis syns.

Exempel med HSV-varden direkt i kommandot:

```bash
/home/johannes/projects/oak_camera/.venv/bin/python \
  scripts/oakdsr_yolo_seg_localizer.py --web-host 0.0.0.0 \
  --width 1280 --height 800 \
  --stereo-width 1280 --stereo-height 800 \
  --fps 2 --camera-rotation 180 \
  --viewer-interval-ms 1000 --viewer-scale 0.75 \
  --depth-mask hsv --hsv-low 62,35,35 --hsv-high 158,255,255 \
  --hsv-open-kernel 0 --hsv-close-kernel 2
```

Exempel med sparad HSV-konfiguration fran `oak_camera`:

```bash
/home/johannes/projects/oak_camera/.venv/bin/python \
  scripts/oakdsr_yolo_seg_localizer.py --web-host 0.0.0.0 \
  --width 1280 --height 800 \
  --stereo-width 1280 --stereo-height 800 \
  --fps 2 --camera-rotation 180 \
  --viewer-interval-ms 1000 --viewer-scale 0.75 \
  --depth-mask hsv \
  --hsv-config /home/johannes/projects/oak_camera/configs/hsv_filter.json
```

Om HSV-masken ger for fa giltiga depth-pixlar faller scriptet som standard
tillbaka till vanlig segmenteringsmask. Det syns i terminalen som
`source=seg-depth-fallback`. Med `--no-hsv-fallback` kan fallback stangas av
for renare felsokning.

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
- `--depth-mask`: `segmentation` eller `hsv`. Standard `segmentation`.
- `--hsv-config`: JSON-fil med sparade HSV-varden.
- `--hsv-low`, `--hsv-high`: HSV-granser direkt i kommandot, format `H,S,V`.
- `--hsv-open-kernel`, `--hsv-close-kernel`: morfologi pa HSV-masken.
- `--no-hsv-fallback`: kraver HSV-depth och faller inte tillbaka till
  segmenteringsmask.
- `--smooth-window`: medianutjamning av `x,y,z`. Standard `1`.
- `--viewer-interval-ms`: hur ofta browserbilden uppdateras. Standard
  `1000`.
- `--no-grid-transform`: visar bara `cam` och hoppar over kalibrerad
  `grid`-koordinat.

## Ny segmenteringsdataset

Nar en ny modell ska tranas samlas forst bilder med OAK-D SR och annoteras med
polygonmasker i Label Studio.

Stoppa forst eventuell korande localizer, eftersom bara en process kan aga
kameran at gangen. Starta sedan capture-scriptet:

```bash
/home/johannes/projects/oak_camera/.venv/bin/python \
  scripts/oakdsr_capture_segmentation_dataset.py --web-host 0.0.0.0 \
  --class-name knopp --split train \
  --width 1280 --height 800 --fps 2 --camera-rotation 180
```

Oppna:

```text
http://johannesmelin.local:8095
```

Terminalkommandon:

- `c` + Enter: spara en bild.
- `b` + Enter: spara en burst, standard 5 bilder.
- `g` + Enter: spara bakgrundsbild utan objekt.
- `q` + Enter: avsluta.

Bilderna sparas i:

```text
segmentation_dataset/images/<split>/<klass>/
```

och metadata i:

```text
segmentation_dataset/metadata.csv
```

Ta en blandning av:

- enskilda tydliga knoppar,
- flera knoppar i samma bild,
- delvis skymda knoppar,
- nagra bilder utan knoppar.

Vid skymning ska bara den synliga delen av varje knopp annoteras. Varje synlig
knopp ska vara en egen polygoninstans.

Skapa darefter Label Studio-importfil med inbaddade bilder:

```bash
/home/johannes/projects/oak_camera/.venv/bin/python \
  scripts/create_label_studio_tasks.py --prepared-dir segmentation_dataset \
  --output segmentation_dataset/label_studio_tasks_embedded.json --embed
```

Importera `segmentation_dataset/label_studio_tasks_embedded.json` i Label
Studio. Labeling setup ska vara polygonbaserad:

```xml
<View>
  <Image name="image" value="$image"/>
  <PolygonLabels name="label" toName="image">
    <Label value="knopp"/>
  </PolygonLabels>
</View>
```

Nar annoteringen ar klar exporteras Label Studio-projektet till YOLO
segmentation-format:

```bash
/home/johannes/projects/oak_camera/.venv/bin/python \
  scripts/export_yolo_seg_from_label_studio.py \
  --prepared-dir segmentation_dataset \
  --output-dir yolo_seg_dataset \
  --project <projektnamn-eller-id> --overwrite
```

Kontrollera annoteringarna med en kontaktkarta:

```bash
/home/johannes/projects/oak_camera/.venv/bin/python \
  scripts/preview_yolo_seg_dataset.py --data yolo_seg_dataset/data.yaml \
  --split val --contact-sheet yolo_preview/val_seg_contact_sheet.jpg
```

Trana modellen:

```bash
/home/johannes/projects/oak_camera/.venv/bin/python \
  scripts/train_yolo_seg.py --data yolo_seg_dataset/data.yaml \
  --epochs 50 --imgsz 960 --batch 1 \
  --name knopp_oakdsr_yolo_seg_960_e50
```

Senaste OAK-D SR-traningen anvande `43` bilder, polygonmasker fran Label
Studio och gav modellen:

```text
training_runs/knopp_oakdsr_yolo_seg_960_e50/weights/best.pt
```
