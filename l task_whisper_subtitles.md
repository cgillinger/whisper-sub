# Whisper Auto-Subtitles for Emby

**Projekt:** emby-whisper  
**Repo:** `cgillinger/emby-whisper`  
**Syfte:** Automatisk undertextgenerering för Emby-biblioteket via Whisper  
**Målmiljö:** server2 (i3-12100, 32GB RAM, Ubuntu 24.04, Docker)

---

## Bakgrund

Emby-servern på server2 servar video från `/mnt/media6tb/emby_media/` (filmer, serier, hemvideor). De flesta videor saknar undertexter. Whisper kan generera undertexter automatiskt — transkribering för svenska, översättning till engelska för allt annat.

**Emby plockar upp externa undertextfiler automatiskt** om de följer namnkonventionen `Filnamn.sv.srt` / `Filnamn.en.srt` och ligger bredvid videofilen. Ingen Emby-plugin behövs.

### Strategi

1. Whisper detekterar språk i de första ~30 sekunderna
2. Svenska → transkribera (behåll svenska) → `.sv.srt`
3. Alla andra språk → translate (Whispers inbyggda översättning till engelska) → `.en.srt`

### Hårdvara

- CPU: i3-12100 (4C/8T) — tillräcklig för bakgrundsbearbetning
- iGPU: Intel UHD 730 — OpenVINO möjlig (redan konfigurerat för Immich)
- RAM: 32 GB — large-v3-modellen kräver ~3–4 GB
- Media: `/mnt/media6tb/emby_media/` (6 TB, ext4)

### Docker-konventioner (server2)

- Stack: `/mnt/docker/stacks/emby-whisper/docker-compose.yml`
- Appdata: `/mnt/docker/appdata/emby-whisper/`
- PUID/PGID: 1000, TZ: Europe/Stockholm
- Modeller cachas i appdata (undvik omhämtning vid rebuild)

---

## Fas 1 — MVP: Transkribera en fil (Sonnet)

### Uppgift

Skapa ett Python CLI-verktyg som tar en videofil som input och genererar en `.srt`-fil bredvid den.

### Krav

- Python 3.10+
- Bibliotek: `faster-whisper` (CTranslate2-backend, effektivare än OpenAI Whisper)
- Modell: `large-v3` (Whispers bästa generella modell)
- Compute: `cpu` (int8-kvantisering för hastighet)
- Input: sökväg till en videofil
- Output: `.srt`-fil i samma katalog som videofilen

### Logik

```
1. Ladda modellen (large-v3, device=cpu, compute_type=int8)
2. Kör language detection på filen (första 30s)
3. Om detekterat språk == "sv":
     → task="transcribe", language="sv"
     → spara som Filnamn.sv.srt
4. Annars:
     → task="translate" (till engelska, inbyggt)
     → spara som Filnamn.en.srt
5. Generera SRT-format med korrekt tidsstämpling
```

### SRT-format

```
1
00:00:01,000 --> 00:00:04,500
This is the first subtitle line.

2
00:00:05,000 --> 00:00:08,200
This is the second subtitle line.
```

Varje segment från faster-whisper har `start`, `end`, `text`. Konvertera till SRT med sekventiellt index, tidsstämplar i `HH:MM:SS,mmm`-format.

### Filstruktur

```
emby-whisper/
├── README.md
├── whisper_sub.py          ← CLI-verktyget
├── requirements.txt        ← faster-whisper
└── .gitignore
```

### CLI-interface

```bash
python whisper_sub.py /path/to/video.mp4
# → skapar /path/to/video.sv.srt eller /path/to/video.en.srt

python whisper_sub.py /path/to/video.mp4 --dry-run
# → skriver ut detekterat språk och planerad åtgärd, ingen fil skapas

python whisper_sub.py /path/to/video.mp4 --model small
# → använd en mindre modell (snabbare, sämre kvalitet)
```

### Argument

| Argument | Default | Beskrivning |
|----------|---------|-------------|
| `path` | (krävs) | Sökväg till videofil |
| `--model` | `large-v3` | Whisper-modell |
| `--device` | `cpu` | `cpu` eller `cuda` |
| `--compute-type` | `int8` | Kvantisering (`int8`, `float16`, `float32`) |
| `--dry-run` | false | Detektera språk, skriv inte fil |
| `--force` | false | Skriv över befintlig .srt |

### Verifieringschecklista

- [ ] `python whisper_sub.py testfil.mp4` genererar en `.srt`-fil
- [ ] SRT-filen har korrekt format (index, tidsstämplar, text)
- [ ] Svensk video → `.sv.srt`
- [ ] Engelsk video → `.en.srt`
- [ ] `--dry-run` skriver ut språk utan att skapa fil
- [ ] Befintlig `.srt` skippas (utan `--force`)

---

## Fas 2 — Batch-skanning och kö (Sonnet)

### Uppgift

Lägg till ett `scan`-kommando som skannar en katalogstruktur, hittar videor utan undertexter och bearbetar dem sekventiellt.

### Krav

- Skanna rekursivt efter videofiländelser: `.mp4`, `.mkv`, `.avi`, `.m4v`, `.mov`, `.wmv`, `.ts`
- Hoppa över filer som redan har en matchande `.sv.srt` eller `.en.srt`
- Bearbeta en fil i taget (ingen parallelism — CPU-bunden)
- State-fil (JSON) som spårar bearbetade filer → undvik omarbetning vid omstart
- Logga progress: filnamn, detekterat språk, tid, resultat

### CLI-tillägg

```bash
python whisper_sub.py scan /mnt/media6tb/emby_media/
# → skannar rekursivt, transkriberar allt utan befintliga undertexter

python whisper_sub.py scan /mnt/media6tb/emby_media/ --dry-run
# → listar filer som SKULLE bearbetas, med uppskattad tid

python whisper_sub.py scan /mnt/media6tb/emby_media/ --limit 5
# → bearbeta max 5 filer (för testning)
```

### State-fil

Sparas i `--state-file` (default: `~/.emby-whisper-state.json`).

```json
{
  "processed": {
    "/mnt/media6tb/emby_media/Emby/Film.mkv": {
      "language": "en",
      "task": "translate",
      "output": "/mnt/media6tb/emby_media/Emby/Film.en.srt",
      "timestamp": "2026-04-08T14:30:00",
      "duration_seconds": 342.5,
      "model": "large-v3"
    }
  },
  "errors": {
    "/mnt/media6tb/emby_media/Emby/Broken.avi": {
      "error": "Could not decode audio",
      "timestamp": "2026-04-08T15:00:00"
    }
  }
}
```

### Verifieringschecklista

- [ ] `scan` hittar alla videofiler rekursivt
- [ ] Filer med befintlig `.srt` hoppas över
- [ ] State-fil skapas och uppdateras efter varje fil
- [ ] Omstart av scan fortsätter där det slutade
- [ ] `--limit` begränsar antal bearbetade filer
- [ ] `--dry-run` listar filer utan bearbetning
- [ ] Fel i en fil stoppar inte resten av kön

---

## Fas 3 — Docker-container (Sonnet)

### Uppgift

Paketera som Docker-container med stöd för schemalagd körning och persistent modellcache.

### Docker-struktur

```
/mnt/docker/stacks/emby-whisper/
└── docker-compose.yml

/mnt/docker/appdata/emby-whisper/
├── config.yml              ← konfiguration
├── state.json              ← bearbetningshistorik
└── models/                 ← cachat modelldata (persistent)
```

### docker-compose.yml

```yaml
services:
  emby-whisper:
    build:
      context: /mnt/docker/appdata/emby-whisper/app
    container_name: emby-whisper
    environment:
      - PUID=1000
      - PGID=1000
      - TZ=Europe/Stockholm
    volumes:
      # Media (read-write — skriver .srt-filer bredvid videor)
      - /mnt/media6tb/emby_media:/media
      # Appdata
      - /mnt/docker/appdata/emby-whisper/config.yml:/app/config.yml:ro
      - /mnt/docker/appdata/emby-whisper/state.json:/app/state.json
      # Modellcache (undvik omhämtning vid rebuild)
      - /mnt/docker/appdata/emby-whisper/models:/root/.cache/huggingface
    # Temperatursensorer (read-only)
      - /sys/class/thermal:/sys/class/thermal:ro
      - /sys/class/hwmon:/sys/class/hwmon:ro
    deploy:
      resources:
        limits:
          cpus: "3.0"    # max 3 av 4 kärnor — lämna marginal för OS + andra containers
    restart: "no"
```

> **Viktigt:** Containern behöver read-write-access till mediamappen för att skriva `.srt`-filer. Inget annat skrivs till mediamappen.

### config.yml

```yaml
# Kataloger att skanna (container-sökvägar)
scan_paths:
  - /media/Emby
  - /media/EgenFilm1

# Whisper-inställningar
model: large-v3
device: cpu
compute_type: int8

# Språklogik
swedish_threshold: 0.7    # minsta confidence för att räkna som svenska
default_task: translate    # om ej svenska → översätt till engelska

# Filtyper
video_extensions:
  - .mp4
  - .mkv
  - .avi
  - .m4v
  - .mov

# Beteende
skip_if_subtitle_exists: true
max_files_per_run: 0       # 0 = obegränsat

# Termisk throttling
thermal:
  enabled: true
  cpu_pause_temp: 72       # pausa kön om CPU >= 72°C
  nvme_pause_temp: 70      # pausa kön om NVMe >= 70°C
  resume_temp: 65           # återuppta när BÅDA < 65°C
  check_interval: 30        # sekunder mellan temperaturkontroller under transkribering
  max_pause_minutes: 30     # avbryt filen om pausad längre än 30 min (väntläge indikerar problem)
```

### Körning

Containern startar, skannar, bearbetar kön och avslutar (`restart: no`).

Schemaläggning via cron på host:

```bash
# /etc/cron.d/emby-whisper
0 8 * * * root cd /mnt/docker/stacks/emby-whisper && docker compose up --build 2>&1 | logger -t emby-whisper
```

Körs kl 08:00 dagligen (efter boot 07:00, innan shutdown 03:00 — 19h fönster).

### Termisk throttling — implementation

Containern läser temperaturer från bind-monterade sysfs-sökvägar.

**CPU-temperatur:** `/sys/class/thermal/thermal_zone*/temp` — hitta zonen som heter `x86_pkg_temp` eller `coretemp`. Värdet är i milligrader (t.ex. `72000` = 72°C).

**NVMe-temperatur:** `/sys/class/hwmon/hwmon*/temp1_input` — identifiera NVMe via `hwmon*/name` som innehåller `nvme`. Samma milligradformat.

**Logik:**
1. Innan varje ny fil i kön: kontrollera temperatur
2. Under transkribering: kontrollera var 30:e sekund (i en separat tråd eller mellan segment)
3. Vid överskridande: logga "Thermal pause: CPU 73°C (threshold 72°C)" → vänta i loop med 30s intervall
4. Vid återupptagande: logga "Thermal resume: CPU 64°C, NVMe 58°C"
5. Om pausad > `max_pause_minutes`: avbryt aktuell fil, logga som error, fortsätt med nästa

### Verifieringschecklista

- [ ] `docker compose up --build` bygger och startar containern
- [ ] Modeller cachas i `/mnt/docker/appdata/emby-whisper/models/` och överlever rebuild
- [ ] State-fil persistent mellan körningar
- [ ] `.srt`-filer skrivs till rätt plats i mediamappen
- [ ] Container avslutar rent efter avslutad kö (exit code 0)
- [ ] Emby hittar och visar de genererade undertexterna i UI
- [ ] CPU begränsad till 3 kärnor (`docker stats` visar max ~300% CPU)
- [ ] Termisk throttling pausar vid CPU ≥ 72°C eller NVMe ≥ 70°C (testa med lägre tröskel)
- [ ] Kön återupptas automatiskt när temperaturen sjunker under 65°C
- [ ] Termisk paus loggas tydligt (temperatur, sensor, varaktighet)

---

## Fas 4 — KB-Whisper för bättre svenska (Sonnet)

### Uppgift

Lägg till stöd för KBLab/whisper-large-v3-swedish som alternativ modell vid svensk transkribering.

### Bakgrund

KB-Whisper är finskuren på SVT-undertexter och ger märkbart bättre svenska än vanilla Whisper. Den finns på Hugging Face som `KBLab/whisper-large-v3-swedish`.

`faster-whisper` kan ladda HF-modeller direkt (konverterar till CTranslate2-format vid första laddning, cachas sedan).

### Ändrad logik

```
1. Detektera språk med standard large-v3 (snabbare, pålitligare detection)
2. Om svenska:
     → ladda KB-Whisper-modellen
     → task="transcribe", language="sv"
3. Om annat språk:
     → använd standard large-v3
     → task="translate"
```

### Konfiguration

```yaml
# config.yml — tillägg
swedish_model: KBLab/whisper-large-v3-swedish   # modell för svensk transkribering
default_model: large-v3                          # modell för allt annat
```

### OBS: Minneskonsumtion

Två large-modeller i minne samtidigt kräver ~6–8 GB RAM. Alternativ:
- Ladda en modell i taget (långsammare vid modellbyte, men säkrare)
- Kör hela svenska kön först, sedan resten

Rekommendation: **En modell i taget.** Sortera kön efter detekterat språk — alla svenska först med KB-Whisper, sedan alla andra med large-v3. En modell-laddning per batch.

### Verifieringschecklista

- [ ] Svensk video transkriberas med KB-Whisper-modellen
- [ ] Icke-svensk video använder standard large-v3
- [ ] KB-Whisper-modellen cachas i models-katalogen
- [ ] Minnesanvändning håller sig under ~5 GB (en modell i taget)
- [ ] Kvalitetsjämförelse: kör samma svenska video med båda modellerna, jämför output

---

## Fas 5 — OpenVINO-acceleration (Opus)

> **Opus rekommenderas** — OpenVINO-integration med faster-whisper är icke-trivial, kräver rätt modellkonvertering och device-konfiguration, och delar iGPU med Immich ML.

### Uppgift

Aktivera Intel OpenVINO som compute-backend för faster-whisper, för att utnyttja iGPU (UHD 730) och få 2–4x snabbare transkribering.

### Bakgrund

- `faster-whisper` stödjer OpenVINO via `device="openvino"`
- iGPU redan konfigurerad: `/dev/dri/renderD128` tillgänglig, Immich ML kör OpenVINO
- Potentiell konflikt: Immich ML och emby-whisper delar iGPU — men Immich ML körs bara vid nya bilder, emby-whisper körs schemalagt, så kollision osannolik

### Ändringar i docker-compose.yml

```yaml
services:
  emby-whisper:
    # ... befintlig config ...
    devices:
      - /dev/dri/renderD128:/dev/dri/renderD128
    group_add:
      - "video"
      - "render"
```

### Ändringar i config.yml

```yaml
device: openvino          # ändrat från "cpu"
compute_type: int8        # OpenVINO stödjer int8
```

### Kända utmaningar

1. **OpenVINO-modellkonvertering:** faster-whisper behöver modellen i OpenVINO IR-format. Konvertering sker automatiskt vid första laddning, men kan ta tid och kräver specifika OpenVINO-versioner.

2. **KB-Whisper + OpenVINO:** Okänt om KBLab-modellen fungerar direkt med OpenVINO-backend. Kan kräva manuell konvertering. CPU-fallback ska fungera automatiskt.

3. **iGPU-delning:** Om Immich ML och emby-whisper kör samtidigt → prestandaförlust men ingen krasch. Schemalägg så de inte överlappar (emby-whisper dagtid, Immich ML nattetid).

4. **Docker-image:** Behöver OpenVINO runtime i imagen. Antingen:
   - Basera på `openvino/ubuntu22_runtime` och installera Python/faster-whisper ovanpå
   - Eller installera `openvino` pip-paket i befintlig image

### Verifieringschecklista

- [ ] Container startar med `/dev/dri/renderD128` monterad
- [ ] `device=openvino` fungerar utan krasch
- [ ] Transkribering av testfil mätbart snabbare än CPU
- [ ] Standard large-v3 fungerar med OpenVINO
- [ ] KB-Whisper fungerar med OpenVINO (eller faller tillbaka till CPU graciöst)
- [ ] Immich ML påverkas inte negativt vid samtidig körning
- [ ] Modellcache fungerar (ej omkonvertering vid varje start)

---

## Fas 6 — Hemvideo-anpassning (Sonnet)

### Uppgift

Anpassa för hemvideor i `/media/EgenFilm1/` som kan ha låg ljudkvalitet, bakgrundsljud, och blandade språk.

### Tillägg

- `--vad-filter true` (Voice Activity Detection) — hoppa över tystnad, minskar hallucinationer
- `--min-silence-duration 0.5` — justera VAD-känslighet
- Konfigurerbar confidence-tröskel per katalog i `config.yml`:

```yaml
scan_paths:
  - path: /media/Emby
    min_confidence: 0.7
  - path: /media/EgenFilm1
    min_confidence: 0.5      # lägre tröskel för hemvideor
    vad_filter: true
    default_language: sv      # anta svenska om osäker
```

### Verifieringschecklista

- [ ] Hemvideo med bakgrundsljud transkriberas utan hallucinationer
- [ ] VAD-filter aktivt för EgenFilm1
- [ ] `default_language` respekteras vid låg confidence
- [ ] Korta videor (<10s) hanteras korrekt

---

## Allmänna riktlinjer för alla faser

### Git

- Commitmeddelanden på engelska, imperativ form
- `.gitignore`: `__pycache__/`, `*.pyc`, `.env`, `models/`, `state.json`
- Inga modeller eller state-filer committade

### Kodstandard

- Python 3.10+, typannotationer
- `argparse` för CLI
- Logging via `logging`-modulen (inte print)
- Docstrings på alla publika funktioner

### Felhantering

- En trasig fil får aldrig stoppa hela kön
- Timeout per fil (konfigurerbart, default 2h)
- Graceful shutdown vid SIGTERM (spara state)

### README.md

- Engelska, professionell ton (self-hosted/homelab-community)
- Badges: Python-version, license
- Tydlig installationssektion
- Exempel på output i Emby UI (screenshot-placeholder)
