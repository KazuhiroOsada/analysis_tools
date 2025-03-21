Analysis tools for GEMSIS-RC
GitHub:  

[Last update]　2025/03/31

・はじめに
    GEMSIS-RCで出力されたファイルを読み込むためのPythonスクリプト
    実行にはnumpy, matplotlibをインストールしたPython3が必要
    スレッド並列によって高速なファイル読み込みが可能


・ファイル構成
    base.py
        -> Run(各ランでの設定・結果の格納)
    coordinate.py
        -> ModifiedDipole(グリッドの計算・描画), VectorTransformer(ダイポール座標ベクトルの変換)
    extract.py
        -> 読み込んだデータの書き出し
    reader.py
        -> DataReader(ファイルを読み込んで、base.Runに代入)
    snapshot.py
        -> 計算結果の赤道面でのスナップショットを作成する


・使い方



・extract.pyの出力形式
    ===========================
    Binary Data Format for Grid
    ===========================
        - Number of species : 4byte integer
        - Array dimensions  : 4byte integer (5,) => N3, N2, N1, Nm, Nv
        - Gridpoints        : 8byte real (3, N1, N2, N3)

        for each species
        - Parallel velocity : 8byte real (Nv,)
        - Magnetic moment   : 8byte real (Nm,)

        * Note
        - unit of position is Re
        - unit of velocity is km/s
        - unit of magnetic moment is chosen such that Vperp[km/s] = sqrt(mu*B[nT])
        - array shape above is in column-major order (compatible with Fortran)

    ===========================
    Binary Data Format for PSD
    ==========================
        - Number of species : 4byte integer
        - Array dimensions  : 4byte integer (5,) => N3, N2, N1, Nm, Nv

        for each time step
        - Time               : 8byte real
        - Magnetic Field |B| : 8byte real (N1, N2, N3)
        for each species
            - Phase Space Density : 8byte real (Nv, Nm, N1, N2, N3)

        * Note
        - unit of magnetic field is nT
        - array shape above is in column-major order (compatible with Fortran)

    ====================================================
    Binary Data Format for 3D Visualization using Mayavi
    ====================================================
        - Array dimensions : 4-byte integer (3,) => N3+1, N2, N1
        - Gridpoints       : 8-byte real (3, N1, N2, N3+1)

        for each time step
        - Time   : 8-byte real
        - Moment : 8-byte real (4, N1, N2, N3+1)
        - B      : 8-byte real (3, N1, N2, N3+1)
        - V      : 8-byte real (3, N1, N2, N3+1)
        - J      : 8-byte real (3, N1, N2, N3+1)

        * Unit
        - Coordinate : Re
        - Density    : cm^-3
        - Pressure   : nPa
        - B          : nT
        - V          : km/s
        - J          : nA/m^2

        * Note
        - order of moment array is N, Vpara, Ppara, Pperp
        - vectors are in the dipole coordinate unless -v option is given
        - B includes the background dipole
        - array shape above is in column-major order (compatible with Fortran)


・履歴
    2025/03/31 作成