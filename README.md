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
  - このとき拡張子だけ異なるファイル名にする
- `data`ディレクトリに配置する

### alignment
`align.sh`を実行し、アライメントファイルを作成する
[fast_align](https://github.com/clab/fast_align)と[mosesdecoder](https://github.com/moses-smt/mosesdecoder)を事前にインストールしておき、そのパスを指定する必要がある

### preprocess
`preprocess.sh`を実行し、データセットの前処理を行う

- preprocess.py
  - 使用するデータの前処理を行うスクリプト
- preprocess.sh
  - preprocess.pyを実行するためのシェルスクリプト
  - データの指定や保存先などを指定して実行する
  - 以下はわかりやすいようにデータセットごとにスクリプトを分けたもの
    - preprocess_lm_data.sh
    - preprocess_noise_data.sh
    - preprocess_test.sh
    - preprocess_train.sh


### generate pseudo data
`noise.sh`を実行し、擬似誤りデータセットを生成する


### pretrain


### train


### generate
