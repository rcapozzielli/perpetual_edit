# perpetual_edit

Observa a webcam em tempo real, detecta rosto, expressões e gestos com MediaPipe e, quando reconhece um gatilho, toca um som de pré-edit e sobrepõe um clipe (vídeo + áudio) estilo TikTok por cima da câmera. Depois volta a esperar o próximo gatilho.

```
WAITING ──gatilho──▶ PRE_EDIT (som curto + rótulo) ──▶ PLAYING (clipe) ──▶ WAITING
```

Na tela: HUD de câmera de vigilância (LIVE, REC), um cubo 3D verde em arame em volta da cabeça, que gira com a pose (yaw/pitch do MediaPipe e a inclinação pelos olhos), e a janela do edit no canto (modo `slot`, veja `overlay_mode`).

## Inspiração

O projeto nasceu [deste vídeo do TikTok, de @mundane.mp4](https://www.tiktok.com/@mundane.mp4/video/7678133596951006486). Com a legenda "my confidence booster app 🔥", ele mostra um notebook com a webcam aberta e uma caixa verde em volta do rosto. Uma janela no canto fica esperando, com uma mira. Quando a pessoa faz alguma coisa, como tirar os óculos ("off glasses"), um edit toca dentro dessa janela.

O `perpetual_edit` é uma recriação independente da ideia, feita do zero. Não tem ligação com o autor do vídeo, e o vídeo não está neste repositório.

## O que vem no repositório

Vêm o código, os templates em JSON (só tempos e efeitos) e as ferramentas. **Não vêm** gravações, amostras de rosto, modelos treinados nem os vídeos dos templates, que são de terceiros.

Por isso, logo depois de clonar, os templates ficam indisponíveis e os gatilhos tocam os clipes de teste gerados por `tools\make_placeholders.py`. O app avisa isso no início. Para usar os templates, coloque os seus próprios vídeos em `capcut_video\` (veja [Adicionar templates novos](#adicionar-templates-novos-importador)) ou adicione [seus próprios clipes](#adicionar-seus-próprios-clipes).

## Instalação (Windows)

Precisa de Python 3.11+ (testado com 3.14). Não precisa instalar ffmpeg: o pacote `imageio-ffmpeg` já traz um.

```powershell
git clone https://github.com/rcapozzielli/perpetual_edit.git
cd perpetual_edit
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python tools\make_placeholders.py   # gera clipes e sons de teste
python main.py
```

Na primeira execução os modelos do MediaPipe (`face_landmarker.task`, `gesture_recognizer.task`) são baixados para `models/`.

## Controles

| Tecla | Ação |
|---|---|
| Q / Esc | sair |
| D | modo debug: landmarks do rosto e das mãos, yaw/pitch, gesto, blendshapes mais fortes e, no painel, `valor/limiar` e `hold atual/necessário` de cada gatilho |
| R | recarregar o `config.json` sem reiniciar |
| Espaço | dispara um edit aleatório na hora |
| L | **temporário:** dispara o template `lossing` (tela verde) na hora |
| G | começa/para a gravação da sessão em `recordings/sessao_<data>.mp4` |
| T | entra/sai do modo treino (gatilhos personalizados) |
| E | abre o **menu de gatilhos**: W/S ou setas movem, ESPAÇO liga/desliga o selecionado, A liga/desliga todos, E ou Esc fecha |
| H | esconde/mostra o painel lateral (a janela fica só com a câmera) |

O que você escolhe no menu (E) e no painel (H) fica salvo em `custom/state.json` e vale de novo quando abrir o app. O `config.json` não é alterado. Um gatilho com `"enabled": false` no config continua desligado; no menu ele aparece marcado como `config`.

## Como funciona o player com áudio

Os frames do clipe são lidos com OpenCV. O áudio do trecho `start→end` é extraído **uma vez** para `cache/*.wav` (com o ffmpeg do `imageio-ffmpeg`) e tocado com o `pygame`. No instante em que o áudio começa guardamos `t0`; a cada frame da câmera mostramos o frame do clipe correspondente a `(agora − t0) × fps`, pulando ou repetindo frames. Assim o áudio nunca engasga e o vídeo acompanha, mesmo se a câmera cair para 20 fps.

Os clipes de teste piscam em branco e dão um bipe a cada segundo inteiro: se o flash e o bipe coincidem, a sincronia está boa.

## Adicionar seus próprios clipes

1. Coloque o arquivo em `videos/` (mp4, mov, webm... qualquer coisa que o ffmpeg leia).
2. Declare o edit em `"edits"` no `config.json`:
   ```json
   "meu_edit": { "file": "videos/meu_clipe.mp4", "start": "00:12.30", "end": "00:16.00" }
   ```
   `start`/`end` aceitam `ss.xx`, `mm:ss.xx` ou `hh:mm:ss.xx`. Sem `end`, toca até o fim.
3. Ligue o edit a um ou mais gatilhos em `"edits": ["meu_edit", ...]`. Com vários, o app sorteia sem repetir o último.
4. Aperte **R** no app.

## Templates com a sua imagem (ex.: template do CapCut)

Um **template** recria um edit usando a **sua câmera** no lugar das filmagens de exemplo. O app usa a música e os tempos do edit original e aplica os efeitos em tempo real. O template do CapCut em si não pode ser aberto pelo Python (o formato é fechado), por isso a recriação parte da **prévia exportada** em mp4.

### O fluxo
```
gesto ──▶ gatilho ──▶ LOADING (2 s: som + barra "LOADING EDIT") ──▶ edit feito com o replay
          │◀── 1,5 s antes ──▶│◀──────── 2 s depois ────────▶│
          └──────── replay capturado (~3,5 s de você fazendo o gesto) ──────┘
```
O loading serve de aviso e também dá tempo de gravar mais do gesto. A tela mostra "CAPTURANDO REPLAY - SEGURA O GESTO". Quando o edit começa, o replay é congelado, e o edit inteiro usa só essas imagens já gravadas.

### Templates incluídos
**`capcut_1031`** (`capcut_video/1031.mp4`, 8,8 s):

| Tempo | Imagem | Efeitos |
|---|---|---|
| 0–2,33 s | replay desde 1,5 s antes do gatilho, velocidade normal | `fade_in`, `vignette`, `face_sticker` 💔 no nariz; no fim a imagem sai para a esquerda (`whip` de saída) enquanto o 😆 entra |
| 2,33–5,03 s | `color` (fundo preto) | emojis recriados: 😆 girando e encolhendo → 😛 → 🧏 acenando |
| 5,03–8,8 s | replay do gesto em **câmera lenta** (0,6×) | `whip` (chicote), `shake` (balanço), `fade_out` |

| Template | Vídeo | Como foi feito |
|---|---|---|
| `capcut_1031` | `1031.mp4` (3:4, 8,8 s) | à mão (tabela acima) |
| `capcut_1031_1` | `1031 (1).mp4` (3:4, 11,2 s) | à mão: intro + 12 cortes rápidos alternando flash e corte seco |
| `capcut_1031_2` | `1031 (2).mp4` (9:16, 10,4 s) | à mão: 4 espaços com vinheta oval e emojis 🫢 ☠ 💀 (as trocas são suaves e o importador não as detecta) |
| `capcut_1031_3` | `1031 (3).mp4` (1:1, 14,9 s) | à mão: intro + 8 espaços, depois os mesmos 8 **espelhados** |
| `capcut_1031_4` | `1031 (4).mp4` (1:1, 16,6 s) | **importador automático**: intro longa + 16 cortes com flash |
| `lossing` | `green/lossing_green.mp4` (1:1, 16,6 s) | **tela verde** (o mesmo template do `1031 (4)`, com letras e efeitos originais): veja "Templates com tela verde" |

Nenhum deles usa a imagem dos mp4, então não há marca d'água. Dos vídeos só vem a música.

### Loading e ajustes por template
Os templates entram sozinhos (veja `@templates` abaixo). Para mudar um específico, declare em `"edits"`:
```json
"capcut_1031": { "template": "templates/capcut_1031.json", "pre_edit_duration": 1.5 }
```
- `pre_edit_duration`: tempo de loading (também vale para clipes normais).
- `pre_edit_sound`: som do loading.
- `"pre_edit": false` desliga o loading.
- O app guarda automaticamente câmera suficiente para o maior (`before_s` + loading) entre os templates.
- O replay é gravado quadrado e cada template recorta dele o seu formato (3:4, 9:16, 1:1) sem distorcer.

### Origens de imagem de cada trecho
- `replay`: imagens já gravadas.
  - `replay_at` diz de que momento começa, **em segundos relativos ao gatilho** (`-1.5` = 1,5 s antes, `0.5` = meio segundo depois).
  - `speed` controla a velocidade (`0.6` = câmera lenta, com mistura de frames para ficar suave).
  - Se pedir além do que foi gravado, segura o último frame.
- `live`: câmera ao vivo. Use se quiser que o edit **termine com o estado atual**.
- `color`: fundo de cor sólida (`"color": [r, g, b]`). Serve de base para emojis e textos animados.
- `video`: frames de um vídeo. Por padrão é o `"video"` do template no mesmo tempo, ou outro tempo com `"from"`. Evite: aparece a marca d'água da prévia.

No template, `"replay": { "before_s": 1.5 }` define quanto antes do gatilho começa a captura.

### Efeitos disponíveis
Todos aceitam `"at"` (início, em segundos, dentro do trecho) e `"duration"`.

| type | Parâmetros |
|---|---|
| `fade_in` / `fade_out` | `duration` |
| `flash` | `strength` (flash branco que se apaga) |
| `mirror` | espelha na horizontal |
| `vignette` | `strength` (0–1) |
| `zoom` | `from`, `to`, `ease: "out"`, `anchor: "face"` ou centro |
| `whip` | `direction` (`up`, `down`, `left`, `right`), `mode` (`in` entra desacelerando, `out` sai acelerando e deixa preto), `distance`, `blur` |
| `emoji` | `emoji` (um caractere só: 🧏 funciona, 🧏‍♀️ não), `keys` (quadros-chave `t`, `x`, `y`, `size`, `rot`, `blur`, `alpha`, interpolados), `spin` (graus/s), `wobble` (balanço em graus) |
| `shake` | `amount`, `rotate` (graus), `freq` |
| `face_sticker` | `emoji`, `landmark` (`nose`, `nose_bridge`, `between_eyes`, `forehead`, `mouth`, `chin` ou o número de um ponto), `scale` (fração da largura do rosto), `offset` |

### Adicionar templates novos (importador)
1. Exporte o template do CapCut e coloque o mp4 em `capcut_video\`. O melhor é a versão com o **mapa de espaços** ("intro clip", "clip 1", "Photo 1"...).
2. Rode:
   ```powershell
   python tools\import_template.py            # importa todos os mp4 novos de capcut_video\
   python tools\import_template.py --dry-run  # só mostra o que faria
   ```
   Para cada vídeo ele:
   - detecta **cortes secos**, **flashes**, fade de entrada e o **preto final**;
   - gera `templates\capcut_<nome>.json`, com a música do vídeo e cada espaço preenchido pelo seu replay;
   - usa espaços longos com o replay inteiro ou em câmera lenta, e cortes rápidos alternando momentos diferentes do gesto;
   - recria flash e vinheta onde o original tem flash, e põe zoom ou balanço variados nos cortes secos.
3. Aperte **R** no app. O template novo já entra no sorteio de quem usa `@templates`.

Proteções do importador:
- **Já importado:** vídeo que já tem template é pulado. `--force` recria, exceto templates com `"hand_tuned": true`, que nunca são sobrescritos.
- **Reexportação:** um vídeo com os mesmos cortes de um template existente é reconhecido e pulado.
- **Nomes:** nunca sobrescreve o template de outro vídeo; se o nome já existe, usa `_b`, `_c`...

Limitações (aí vale ajuste manual no JSON, marcando `"hand_tuned": true`):
- **Transições suaves** (dissolvência, troca entre fundos parecidos) não são detectadas como corte. Exemplo: `capcut_1031_2`.
- **Elementos gráficos** do template (emojis, adesivos, espelhamento) não são reconhecidos. Use os efeitos `emoji`, `face_sticker` e `mirror`.
- **Prévias com filmagem de exemplo** (sem mapa de espaços) quase não têm cortes detectáveis. O importador avisa.
- **Letras de música** que aparecem nos templates não são recriadas.

### Templates com tela verde (o jeito mais fiel)
Com a tela verde, **só os espaços "Photo N" viram você**. O resto continua igual ao template: letras da música, intro, emojis, transições, zoom, filtros e piscadas. Não aparece a marca d'água "Vídeos de amostra", porque o vídeo é o seu export e não a prévia.

1. Gere as imagens verdes (uma vez só):
   ```powershell
   python tools\make_green_slots.py     # green_slots\green_A.png e green_B.png
   ```
   `A` é um verde amarelado e `B` um verde azulado, os dois lisos. Um número ou desenho no verde apareceria por cima de você quando o template escurece a tela.
2. No CapCut, abra o template e **alterne: Photo 1 = A, Photo 2 = B, Photo 3 = A**... Se o template pede filmagens (sem "Photo N"), use os verdes no lugar delas. Exporte.
   A cor só marca **onde** você aparece. A imagem de cada espaço vem do seu replay, com um momento diferente do gesto em cada um.
3. Salve o vídeo em `capcut_video\green\<nome>_green.mp4`. O nome vira o nome do template: `lossing_green.mp4` → `templates\lossing.json`.
4. Importe e aperte **R** no app:
   ```powershell
   python tools\import_template.py --dry-run   # mostra os espaços encontrados
   python tools\import_template.py             # gera o .json
   ```
   Vídeos em `capcut_video\green\` (ou terminados em `_green`) são importados como tela verde automaticamente. Para outro lugar, use `--green`.

Como funciona:
- **Espaços:** o importador procura os trechos verdes. Um espaço termina quando o verde some (corte) ou quando o tom troca de A para B ou de B para A (corte direto, sem pausa). Por isso é preciso alternar. O tom que só "escorrega" num fade não conta como corte, e um flash curto no meio do mesmo verde também não.
- **Momentos do gesto:** cada espaço mostra um momento diferente do replay. Os cortes rápidos usam os momentos em sequência (0, 0,9, 0,3, 1,2 s... depois do gatilho), e o 1º espaço longo mostra o replay inteiro.
- **No app:** cada quadro do vídeo é a base. Os pixels verdes (`source: "keyed"`) são trocados pelo seu replay, enquadrado na caixa que o verde ocupa. Se o template dá zoom ou desliza a foto, você vai junto.
- **Luz:** o app mede quanto o template escureceu ou clareou o verde (fade, flash, vinheta) e aplica isso na sua imagem. `"key": {"light": 0.6}` no JSON controla a força: 1 = igual ao verde, 0 = desliga.
- Os efeitos do trecho (`face_sticker`, `zoom`...) continuam valendo e são aplicados só na sua imagem.

Limitações:
- Filtros de **cor** do template sobre a foto (rosa, roxo) não são copiados, só o brilho.
- **Rotação** e distorção da foto não são copiadas, só a posição e a escala.
- Partes **verdes** do próprio template (um texto verde, por exemplo) também seriam trocadas.
- Os vídeos de `capcut_video\` não devem ir para o GitHub: são do CapCut e de terceiros.

### Variedade: `@templates`
Na lista `edits` de um gatilho:
- `"@templates"` vira **todos** os templates;
- `"@clips"` vira todos os clipes comuns.

O app sorteia sem repetir o último. **Todos os gatilhos** usam `@templates`. Os clipes de teste coloridos (`videos\edit_*.mp4`) saíram do config, mas os arquivos continuam lá caso você queira usar `@clips` ou declarar seus próprios clipes.

### Gravações antigas misturadas (arquivo de replays)
Para ter mais variedade de imagens, cada replay usado num edit é guardado em `replays\<data>\` (~4 MB cada). Nos edits seguintes, cada **corte rápido** tem uma chance (`mix`, padrão 25%) de mostrar um trecho de um replay antigo em vez do atual, sempre no mesmo momento do gesto (`replay_at`). A intro e os trechos longos sempre usam o gesto atual.
- Os trechos antigos são lidos do disco durante o loading, então não travam a imagem.
- Só replays com rosto em pelo menos 60% dos frames são guardados (`min_face_ratio`).
- O arquivo mantém os `max_clips` (30) mais recentes e apaga os mais antigos **dessa pasta**.
- **Ao fechar o app, todos os replays são apagados** (`clear_on_exit`, padrão `true`). Ou seja, a mistura usa só os replays da sessão atual. Se o app cair sem fechar direito, a limpeza acontece quando ele abrir de novo. Só são apagadas as pastas com nome de data criadas pelo app; outros arquivos em `replays\` ficam.

```json
"archive": { "enabled": true, "dir": "replays", "max_clips": 30, "mix": 0.25, "min_face_ratio": 0.6, "clear_on_exit": true }
```
**Privacidade:** isso grava imagens da câmera em disco, suas e de quem aparecer. Para não gravar nada, use `"enabled": false`. Para manter os replays entre sessões, use `"clear_on_exit": false`.

Todo `templates\*.json` vira um edit sozinho (`"auto_templates": true`), com o loading definido em `"template_defaults"`. Para tirar um template do sorteio sem apagar o arquivo, ponha o nome em `"exclude_templates"` (hoje: `capcut_1031_4`, substituído pelo `lossing`). Só declare um template em `"edits"` se quiser valores diferentes para ele.

Atenção: a música e os trechos `video` vêm da prévia, então a marca d'água "Vídeos de amostra" aparece nesses trechos. Para postar, o mais seguro é usar o som pelo próprio TikTok ou CapCut.

## Configuração (`config.json`)

| Campo | O que faz |
|---|---|
| `camera_index` | qual webcam (0, 1, ...). Mudar exige reiniciar |
| `camera_backend` | `msmf` (padrão no Windows), `dshow` ou `any`. Em muitas webcams integradas o `dshow` limita 720p a 10 fps |
| `mirror` | espelha a imagem (modo selfie) |
| `overlay_mode` | `slot` (padrão): uma janela fixa no canto, sempre visível, com uma mira e "WAITING...". No gatilho ela vira "EDITING..." (o loading) e o edit toca dentro dela; às vezes a janela cresce para um tamanho maior. `popup`: o edit aparece como um card que "pula" na tela, em posição sorteada, evitando o rosto. `fullscreen`: ocupa a tela, com fundo desfocado se a proporção for diferente. `pip`: canto fixo |
| `slot` | `corner` (`top_right` ou `top_left`), `width` / `big_width` (largura da janela normal e da grande, fração da tela), `big_chance` (chance de um edit usar a janela grande, padrão 0.33), `aspect` (largura/altura, padrão 1.4), `margin`, `top` (px). Edit com proporção parecida preenche a janela (corta as sobras); muito diferente (ex.: 9:16) aparece inteiro, com fundo desfocado |
| `popup` | `height` (altura do card, fração da tela, padrão 0.55), `tilt_deg` (inclinação máxima sorteada, 0 = reto), `border` (borda branca em px), `margin` |
| `pip` | `scale` (fração da largura), `corner` (`top_left`, `top_right`, `bottom_left`, `bottom_right`), `margin` |
| `global_cooldown_s` | tempo mínimo entre o **fim** de um edit e o próximo disparo (padrão do projeto: 15 s) |
| `archive` | gravações antigas misturadas nos edits (veja "Templates") |
| `smoothing` | 0–1, peso do valor novo na média móvel (menor = mais suave, mais lento) |
| `pre_edit_sound` / `post_edit_sound` | sons antes e depois do clipe |
| `pre_edit_duration` | quanto tempo o som de pré-edit toca antes do clipe entrar |
| `volume.edit` / `volume.sfx` | volumes do clipe e dos efeitos |

Cada gatilho:

| Campo | O que faz |
|---|---|
| `enabled` | liga/desliga |
| `type` | tipo de regra (padrão = nome do gatilho). Ex.: `gesture` |
| `threshold` | limiar do valor medido (barra amarela no painel) |
| `hold_frames` | quantos frames seguidos a condição precisa durar |
| `cooldown_s` | tempo mínimo entre dois disparos **deste** gatilho |
| `label` | texto que aparece na tela ao disparar |
| `edits` | lista de edits possíveis (clipes ou templates) |

Regra extra de robustez: depois de um disparo, um gatilho só pode disparar de novo depois que a condição **desligar** (ex.: fechar a boca). Isso evita disparos em sequência com a mesma pose.

### Tipos de gatilho disponíveis

| type | Valor medido |
|---|---|
| `mouth_open` | blendshape `jawOpen` |
| `brows_up` | média de `browInnerUp`, `browOuterUpLeft`, `browOuterUpRight` |
| `gesture` | confiança do gesto escolhido em `"gesture"`: `Thumb_Up`, `Thumb_Down`, `Closed_Fist`, `Open_Palm`, `Pointing_Up`, `Victory`, `ILoveYou` |
| `eyes_closed` | menor entre `eyeBlinkLeft` e `eyeBlinkRight` (os dois olhos fechados). "Piscada longa" = `hold_frames` alto (20 ≈ 0,7 s) |
| `head_turn` | graus de yaw. `"side"`: `left`, `right` ou `any`, relativo à **tela** (com `mirror: true`, a esquerda da tela é a sua esquerda) |
| `hand_near_face` | 1 = mão encostando ou na frente do rosto, 0 = a uma largura de rosto ou mais |
| `face_lost` | segundos sem rosto na câmera (`threshold` = quantos segundos) |
| `face_back` | dispara quando o rosto volta depois de ficar fora por pelo menos `threshold` segundos |
| `custom` | probabilidade da classe `to` de um modelo treinado por você, logo após a transição `from → to` (veja "Gatilhos personalizados") |

Os gatilhos são avaliados na ordem do `config.json`; se dois ficam prontos no mesmo frame, vence o primeiro. Por isso os gestos vêm antes de `hand_near_face` (um joinha perto do rosto conta como joinha).

## Gatilhos personalizados (modo treino)

Servem para estados que as regras prontas não pegam: "com óculos / sem óculos", "comendo", "de boné", "com a mão na boca"...

### Como funciona
1. Cada amostra é o seu rosto **alinhado pelos olhos** (sempre na mesma posição e escala, 128×128) + os 52 blendshapes daquele frame.
2. Do rosto alinhado sai um descritor **HOG** (histograma da direção das bordas em pequenas células da imagem). Bordas fortes como a armação dos óculos aparecem muito bem nele. Isso é a feature `appearance`. Os blendshapes são a feature `expression`.
3. Um classificador `StandardScaler + LogisticRegression` (scikit-learn) aprende a separar as classes. Treina em segundos e roda em ~1 ms por frame.
4. Ao vivo, o modelo mantém um **estado estável**: a classe que ficou com probabilidade ≥ `stable_prob` por `stable_frames` frames seguidos. O gatilho dispara quando o estado estável **muda** de `from` para `to`, em até `fresh_s` segundos depois da mudança. Perder o rosto por um instante não conta como transição.

### 1a. Criar um modelo pelo app (jeito mais fácil)
1. Aperte **T** (modo treino) e depois **N**.
2. Digite o nome (ex.: `bebendo agua`) e dê ENTER. Acentos não funcionam no campo do OpenCV; o nome vira `bebendo_agua`.
3. Confirme as classes. A sugestão é `normal, bebendo_agua`: ENTER aceita, ou digite outras separadas por vírgula (ex.: `sem_fone, com_fone`).
4. O modelo é criado já selecionado, com um **gatilho próprio** que dispara na transição da 1ª para a 2ª classe e toca os templates.

Esses modelos ficam em `custom\models.json` (o `config.json` não é alterado). Para mudar limiar, rótulo, classes ou desligar o gatilho, edite esse arquivo e aperte **R**. Para apagar um modelo, tire a entrada dele do `models.json` e a pasta `custom\<nome>\`.

Modelos criados pelo app vêm configurados para **mão ou objeto perto do rosto** (beber, pôr fone, comer, falar ao telefone):
- `crop_scale: 1.5`: o recorte é mais aberto e pega orelhas, cabelo e a mão chegando na boca.
- `features: ["appearance", "expression", "hands"]`. A feature `hands` diz onde estão as mãos **em relação ao rosto**: presença, posição do punho e das pontas dos dedos, e a distância até a boca e até cada lado do rosto. Tudo em "larguras de rosto", então não muda se você chegar mais perto ou mais longe da câmera.

No treino, cada grupo de features ganha um peso que equilibra o tamanho dele: sem isso, os 1.764 números de aparência "afogariam" os 28 das mãos.

Dicas para esses estados:
- grave a classe `normal` fazendo coisas variadas (falando, mexendo a mão longe do rosto), para o modelo não achar que qualquer mão é "bebendo";
- grave a ação de verdade (copo ou garrafa na boca, fone nas orelhas) em 2–3 sessões curtas;
- para ações rápidas, como "colocando o fone", o que funciona melhor é treinar o **estado final** (`com_fone`): o gatilho dispara na transição `sem_fone → com_fone`.

### 1b. Declarar o modelo à mão (`config.json`)
```json
"custom_models": {
  "glasses": {
    "classes": ["com_oculos", "sem_oculos"],
    "features": ["appearance"],
    "stable_prob": 0.75, "stable_frames": 8, "fresh_s": 2.0
  }
},
"triggers": {
  "glasses_off": {
    "type": "custom", "model": "glasses", "from": "com_oculos", "to": "sem_oculos",
    "threshold": 0.7, "hold_frames": 3, "cooldown_s": 6,
    "label": "off glasses", "edits": ["edit_2"]
  }
}
```
- `classes`: de 2 a 9 (escolhidas pelas teclas 1–9).
- `features`: `["appearance"]` para coisas visuais (óculos, boné, objeto). `["appearance", "expression"]` quando a expressão também muda (comendo, bocejando). Acrescente `"hands"` quando a mão participa (beber, pôr fone). Para óculos, deixar só `appearance` evita que o modelo "decore" a expressão que você fez enquanto gravava.
- `crop_scale`: 1.0 = rosto justo (padrão, bom para óculos); 1.5 = recorte aberto.
- Sem `"from"`, o gatilho dispara ao entrar em `to` vindo de qualquer estado.

### 2. Gravar amostras
1. Aperte **T**. O painel fica amarelo, e um quadro no canto mostra o rosto alinhado que está sendo salvo.
2. Escolha o modelo com **TAB**. Fique no estado da classe 1 (ex.: de óculos) e aperte **1** para começar a gravar. Mexa um pouco a cabeça, aproxime e afaste, mude a expressão. Aperte **1** de novo para parar. Faça isso em 2 ou 3 sessões curtas (~5 s cada).
3. Mude de estado (tire os óculos) e faça o mesmo com **2**.
4. **TAB** troca de modelo, se houver mais de um.

**Apagar amostras pelo app** (no modo treino, sempre pede confirmação: digite `s` e ENTER; ESC cancela):
- **U** desfaz a última sessão gravada do modelo selecionado (ex.: gravou errado).
- **X** e depois o número da classe (ex.: **X 1**) apagam todas as amostras daquela classe.
- **X 0** apaga o **modelo inteiro** selecionado: amostras, `model.joblib` e o gatilho dele. Para confirmar, digite o nome do modelo. Vale só para modelos criados pelo app (`custom/models.json`); os declarados no `config.json` você tira de lá. Vai para `custom/_lixeira/<modelo>/modelo_inteiro_<data>/`, junto com um `models_entry.json` (a entrada que estava no `custom/models.json`, para restaurar à mão).

Nada é apagado de verdade: os arquivos vão para `custom/_lixeira/<modelo>/<classe>/<data>/`. Para desfazer, mova-os de volta para `custom/<modelo>/<classe>/`. Para liberar espaço, apague a pasta `_lixeira`. O modelo já treinado continua valendo até você apertar ENTER de novo.

Dicas: 50–150 amostras por classe bastam. Grave com a mesma iluminação em que vai usar o app. Se puder, grave um pouco em outros horários ou lugares também. As amostras ficam em `custom/<modelo>/<classe>/`, e dá para apagar imagens ruins direto na pasta.

### 3. Treinar e testar
1. Aperte **ENTER**. O painel mostra o resultado, por exemplo `validação 94% (3 folds por sessão)`.
   - A validação separa por **sessão de gravação**: frames da mesma sessão nunca ficam ao mesmo tempo no treino e no teste. Frames seguidos são quase idênticos, e misturar inflaria a acurácia. Por isso só aparece com 2+ sessões por classe.
2. Ainda no modo treino, as barras de cada classe mostram a probabilidade ao vivo, e "estado estável" mostra o que o modelo acha agora. Tire e coloque os óculos e veja se ele acompanha.
3. Se errar muito: grave mais sessões da classe que ele confunde e aperte ENTER de novo (treina com todas as amostras).
   - O painel mostra, ao lado de cada classe, `amostras` e `sessões` (em amarelo quando há só 1). Ao treinar, o app avisa se alguma classe tem 1 sessão só ou se uma classe tem mais que o dobro de amostras da outra.
   - **Dispara toda hora?** Quase sempre o problema é a classe "normal" gravada parada demais. O modelo aprende que "normal" é *exatamente aquela pose*, e qualquer coisa diferente (virar a cabeça, coçar o rosto, outra luz) vira a outra classe. Grave o normal em 3–4 sessões, cada uma de um jeito: falando, virando a cabeça, mão no queixo, mexendo no celular, mais perto e mais longe. Na classe da ação, grave só enquanto a ação acontece (ex.: só com a garrafa na boca).
4. Aperte **T** para sair. O gatilho passa a funcionar.

O modelo fica salvo em `custom/<modelo>/model.joblib` e é carregado automaticamente ao abrir o app (ou com **R**).

## Gravar a sessão para postar

Aperte **G** para começar e **G** de novo para parar (ou feche o app, que ele termina a gravação sozinho). O vídeo sai em `recordings/` em 1280x720, H.264 + AAC, com o HUD (LIVE/REC), os clipes e todos os sons. O painel lateral e o overlay de debug **não** entram no vídeo.

O áudio do vídeo é montado a partir dos sons que o app tocou (clipes e efeitos), cada um posicionado no instante exato. O microfone não é gravado. Se você vai colocar um som em alta pelo próprio TikTok, é só baixar o volume do vídeo lá.

## Calibrar limiares

1. Rode o app e aperte **D**.
2. No painel, cada gatilho mostra a barra do valor atual, a linha amarela do limiar e `valor/limiar hold atual/necessário`.
3. Faça a expressão e veja até onde a barra vai; faça cara neutra e veja onde ela fica. Coloque o `threshold` mais ou menos no meio.
4. Se dispara sem querer, aumente `hold_frames` ou o `threshold`. Se demora para responder, diminua `hold_frames` ou aumente `smoothing`.
5. Salve o `config.json` e aperte **R**.

## Problemas comuns

- **FPS baixo (~10):** confira o `camera_backend` (use `msmf`) e a iluminação; com pouca luz a webcam aumenta a exposição e entrega menos frames.
- **Rosto não detectado:** o detector do Face Landmarker é de curto alcance (até ~2 m). Fique de frente para o notebook; rostos pequenos no canto da imagem não são detectados.
- **Teste sem câmera:** `python tests\test_core.py` verifica config, gatilhos (com valores simulados), player e detector.

## Estrutura

```
main.py                    loop principal e máquina de estados
config.json
perpetual_edit/
  config.py                leitura/validação do config
  capture.py               webcam numa thread (sempre o frame mais novo)
  models.py                download dos modelos .task
  detection.py             Face Landmarker + Gesture Recognizer, suavização
  triggers.py              regras, hold, cooldown, escolha do edit
  audio.py                 mixer pygame, extração/cache do áudio
  player.py                player sincronizado e composição fullscreen/pip
  hud.py                   HUD, painel lateral e overlay de debug
  recorder.py              gravação da sessão (vídeo + mixagem dos sons)
  training.py              modo treino: coleta, alinhamento, HOG, treino e inferência
  template.py              templates: replay, enquadramento no rosto, efeitos e player
  archive.py               arquivo de replays antigos (gravar, sortear, carregar trechos)
templates/                 templates em JSON (ex.: capcut_1031.json)
custom/                    amostras e modelos dos gatilhos personalizados
replays/                   replays antigos usados para variar os edits (criada sozinha)
tools/make_placeholders.py clipes e sons de teste
tools/import_template.py   importa templates do CapCut (mp4) para templates/*.json (prévia ou tela verde)
tools/make_green_slots.py  gera os verdes A e B (green_slots/) para os templates de tela verde
```

## Licença

[MIT](LICENSE).
