# Analysis Tools for GEMSIS-RC

GitHub: [https://github.com/KazuhiroOsada/analysis_tools](https://github.com/KazuhiroOsada/analysis_tools)

**[Last update] 2025/02/21**

---

## はじめに

GEMSIS-RCで出力されたファイルを読み込むためのPythonスクリプト。

- 実行には `numpy`, `matplotlib` をインストールした Python3(3.6以上) が必要
- ファイル読み込みは、concurrent.futures ThreadPoolExecutorによってスレッド並列化
- `analysis/`以下は波動などの解析スクリプト置き場

---

## 主なファイル構成

- `base.py` → `Run`（各ランでの設定・結果の格納
- `chunk_reader.py` → ドメイン単位でデータ読み込み
- `coordinate.py` → `ModifiedDipole`（グリッドの計算・描画）、`VectorTransformer`（ダイポール座標ベクトルの変換）
- `draw.py` → 描画関数
- `equatorial_reader.py` → 赤道面データの読み込み
- `extract.py` → 読み込んだデータの書き出し
- `reader.py` → `DataReader`（ファイルを読み込んで、`base.Run` に代入）
- `snapshot.py` → 計算結果の赤道面でのスナップショットを作成

---

## 使い方

### `base.py`

```python
from base import Run

run = Run('run1')
run.read('coord') # run1/coord**.datを読み込む
run.set_trange((0, 2161, 20)) # 読み込みの時間幅を設定
run.read('field') # run1/field**.datを読み込む　(100sごと)
```

### `extract.py`

例）100 sごとにデータを書き出す:

```bash
python3 extract.py -r rundir -o out.dat -t 0 2161 20 -x mayavi
```

### `snapshot.py`

例）1000 sごとにプロットを作成する:

```bash
python3 snapshot.py -r rundir -t 0 2161 200 #　デフォルトの出力ディレクトリはfigure
python3 snapshot.py -r rundir -o outdir -t 0 2161 200 # 出力ディレクトリを指定する
```

---

## `extract.py` の出力形式

### Binary Data Format for Grid

```markdown
- Number of species : 4-byte integer
- Array dimensions  : 4-byte integer (5,) => N3, N2, N1, Nm, Nv
- Gridpoints        : 8-byte real (3, N1, N2, N3)

for each species:
- Parallel velocity : 8-byte real (Nv,)
- Magnetic moment   : 8-byte real (Nm,)

* Note:
- 位置の単位: Re
- 速度の単位: km/s
- 磁気モーメントの単位は Vperp[km/s] = sqrt(mu*B[nT]) に対応
- 配列はColumn major（as Fortran）
```

### Binary Data Format for PSD

```markdown
- Number of species : 4-byte integer
- Array dimensions  : 4-byte integer (5,) => N3, N2, N1, Nm, Nv

for each time step:
- Time               : 8-byte real
- Magnetic Field |B| : 8-byte real (N1, N2, N3)

for each species:
- Phase Space Density : 8-byte real (Nv, Nm, N1, N2, N3)

* Note:
- 磁場の単位: nT
- 配列はColumn major（as Fortran）
```

### Binary Data Format for 3D Visualization using Mayavi

```markdown
- Array dimensions : 4-byte integer (3,) => N3+1, N2, N1
- Gridpoints       : 8-byte real (3, N1, N2, N3+1)

for each time step:
- Time   : 8-byte real
- Moment : 8-byte real (4, N1, N2, N3+1)
- B      : 8-byte real (3, N1, N2, N3+1)
- V      : 8-byte real (3, N1, N2, N3+1)
- J      : 8-byte real (3, N1, N2, N3+1)

* Unit:
- 座標系: Re
- 密度  : cm^-3
- 圧力  : nPa
- B     : nT
- V     : km/s
- J     : nA/m^2

* Note:
- Moment の順序: N, Vpara, Ppara, Pperp
- ベクトルはデフォルトでダイポール座標系（`-v` オプションで変更可能）
- B はバックグラウンドダイポールを含む
- 配列はColumn major（as Fortran）
```

---

## 履歴

- **2025/04/03**: 作成
- **2025/07/02**: `chunk_reader`の実装
- **2025/09/19**: `analysis/`の追加
- **2026/02/21**: `jgr26`ブランチの追加
