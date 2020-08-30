# transformer-copy

## ディレクトリ構成
- app
  - アプリケーションのソースコード
- corpus_scripts
  - データセットの前処理などのスクリプト
- data
  - 使用するデータセットをここに入れる
- data_art
  - 擬似誤り生成したデータをここに入れる
- dicts
  - 語彙の辞書をここに入れる
  - 実験で使用する分割単位に対応した辞書が必要になる
- out
  - data_bin
    - train,validデータセットのバイナリファイル
  - data_bin_art
    - pseudoデータセットのバイナリファイル
  - data_raw
    - testデータセットのrawファイル
  - log
    - ログファイル
  - models
    - モデルファイル
  - results
    - generateの結果


## 実行手順

### dataset
使用するデータセットを用意する

手順：
- データセットをセグメント(単語分割等)する
- train, valid, testに分割する
- 誤り文側と正解文側でファイルを分ける
- `data`ディレクトリに配置する
- ファイル名の例：
  - lang8.train.src  # 訓練データの誤り文側
  - lang8.train.tgt  # 訓練データの正解文側
  - lang8.valid.src  # 開発データの誤り文側
  - lang8.valid.tgt  # 開発データの正解文側

### alignment
`align.sh`を実行し、アライメントファイルを作成する
[fast_align](https://github.com/clab/fast_align)と[mosesdecoder](https://github.com/moses-smt/mosesdecoder)を事前にインストールしておき、そのパスを指定する必要がある


### generate pseudo data
`noise.sh`を実行し、擬似誤りデータセットを生成する


### preprocess
`preprocess.sh`を実行し、データセットの前処理を行う
訓練データ(train, valid)は`preprocess_train.sh`、評価データは`preprocess_test.sh`のように分けている
前処理されたデータセットは`out/data_bin`もしくは`out/data_raw`に出力される


### pretrain
`pretrain.sh`を実行し、データセット等を指定してpre-trainingを行う
コマンド自体は`train.sh`と同じで、オプションが異なる
学習済みモデルは`out/models`に保存される

### train
`train.sh`を実行し、データセット等を指定してtrainを行う
pretrainモデルを指定してfine-tuningしたい場合は`--pretrained-model $pretrained_model`をオプションに追加する
学習済みモデルは`out/models`に保存される


### generate
`generate.sh`を実行し、学習済みモデルを使って文生成を行う
`data_rows`は評価データのリスト、`epochs`はどのエポックのモデルかのリストで、for文でそれぞれを一度に実行するようにしている
評価データのM2ファイルを用意しておく必要がある。[ERRANT](https://github.com/chrisjbryant/errant)の`errant_parallel`を使用してM2ファイルを作成する。
`python2`を実行できるようにしておく必要がある
生成データは`out/results`に保存される
