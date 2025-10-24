<p align="center">
<strong>=================================================================</strong><br>
<strong>README</strong><br><br>
<strong>日付:</strong> 10/2025<br><br>
<strong>モデレーター:</strong> Daniel.J.Q.Goh<br>
<strong>=================================================================</strong>
</p>

<br><br><br><br>

!!! / この部分には、プロジェクト全体の目標を達成するための主要な目的が含まれています

# Raspberry Pi用リアルタイム人物検出システム

YOLOv11とNCNN最適化を使用したRaspberry Pi向けの、プロフェッショナルでクリーンなリアルタイム人物検出の実装です。

## 機能

**スマート環境検出** - GUIモードまたはヘッドレスモードを自動的に検出  
**YOLOv11 Nanoモデル** - Raspberry Piパフォーマンス用に最適化された最新YOLOバージョン  
**NCNNバックエンド** - より高速な推論のためのハードウェアアクセラレーション  
**デュアルカメラサポート** - USBウェブカメラとPiカメラに対応  
**GUI＆ヘッドレスモード** - ライブ表示または画像保存  
**プロフェッショナルなコード構造** - クリーンで文書化され、保守可能  

## 要件

```bash
pip install ultralytics opencv-python numpy
```

## クイックスタート

### 1. **シンプル実行**
Ultralyticsは、既製のYOLOモデルのダウンロードと使用を簡単にします。これらのモデルはCOCOデータセットで訓練されており、「person」、「car」、「chair」などの80の一般的なオブジェクトを検出できます。**YOLO11n**検出モデルをダウンロードするには、次のコマンドを実行してください：
```bash
yolo detect predict model=yolo11n.pt
```

<div style="page-break-after: always;"></div>

**物体検出モジュール**を実行するには、次のコマンドを実行してプログラムを実行してください：

```bash
python3 obj_detection.py
```
システムは自動的に以下を実行します：
- 環境を検出（GUI/ヘッドレス）
- YOLOモデルの読み込みと最適化
- カメラ接続のテスト
- 最適なモードで検出を開始

### 2. **手動モード選択**

**GUIモード（ウィンドウ表示付き）：**
```python
detector = PeopleDetectionSystem()
detector.run_gui_detection(duration_seconds=0, camera_index=0)
```

**ヘッドレスモード（画像保存）：**
```python
detector = PeopleDetectionSystem()
detector.run_headless_detection(duration_seconds=0, camera_index=0, save_interval=5)
```

**スマート自動検出：**
```python
detector = PeopleDetectionSystem()
detector.run_smart_detection(duration_seconds=0, camera_index=0)
```

## 設定

`config.py`を編集してカスタマイズ：
- カメラ設定（解像度、FPS）
- 検出パラメータ（信頼度閾値）
- 表示オプション（色、フォント）
- ファイル命名規則

<div style="page-break-after: always;"></div>

## 操作方法

### GUIモード：
- **'q'** - 検出を終了
- **'s'** - 現在のフレームを保存
- **ウィンドウを閉じる** - 検出を停止

### ヘッドレスモード：
- **Ctrl+C** - 検出を停止
- 5秒ごとに画像が自動保存
- 終了時に最終フレームを保存

## 出力ファイル

**ヘッドレスモードで生成：**
- `detection_[timestamp]_people_[count].jpg` - 定期保存
- `detection_final_[timestamp].jpg` - 最終フレーム

**GUIモードで生成（'s'キー押下時）：**
- `detection_frame_[timestamp].jpg` - 手動保存

## トラブルシューティング

### カメラウィンドウが開かない：
- **SSHユーザー**: `ssh -X pi@your_pi_ip`で接続
- **ヘッドレスPi**: システムが自動的にヘッドレスモードに切り替え
- **ディスプレイなし**: `sudo apt install python3-opencv libgtk-3-dev`をインストール

### パフォーマンスの問題：
- `config.py`で解像度を下げる
- FPS設定を減らす
- NCNNエクスポートが成功したことを確認

### カメラが見つからない：
- USB接続を確認
- 異なるカメラインデックス（1の代わりに0）を試す
- カメラ権限を確認

<div style="page-break-after: always;"></div>

## システムアーキテクチャ

```
PeopleDetectionSystem (メインクラス)
├── print_system_info()          # システム情報
├── detect_raspberry_pi()        # ハードウェア検出
├── load_model()                 # YOLOv11モデル読み込み & NCNNエクスポート
├── setup_camera()               # カメラ初期化
├── detect_people_in_frame()     # コア検出ロジック
├── check_display_environment()  # 表示機能チェック
├── run_gui_detection()          # ライブ表示付きGUIモード
├── run_headless_detection()     # 画像保存付きヘッドレスモード
├── run_smart_detection()        # 自動モード選択
└── run()                        # メイン実行フロー
```

## パフォーマンス統計

**Raspberry Pi 3B V1.2での典型的なパフォーマンス：**
- **解像度**: 640x480
- **FPS**: 10-15（検出数に依存）
- **モデル**: NCNNを使用したYOLOv11n
- **メモリ**: ~750MB（0.75GB）RAM使用量

## ライセンス

このコードは、教育および開発目的でそのまま提供されています。

## サポート

問題や改善については、以下を確認してください：
1. カメラ接続と権限
2. 表示環境のセットアップ
3. 必要なパッケージのインストール
4. モデルダウンロードの完了
5. モデル訓練中のRaspberry Pi 3B V1.2 CPUとRAMのボトルネック

---

<p align="center">
<em>文書終了</em>
</p>