import glob
import json
import logging
import os
import re
import subprocess
import sys
import time
import traceback
from itertools import chain
from pathlib import Path

# os.system("wget -P cvec/ https://huggingface.co/spaces/innnky/nanami/resolve/main/checkpoint_best_legacy_500.pt")
import gradio as gr
import librosa
import numpy as np
import soundfile
import torch

from compress_model import removeOptimizer
from edgetts.tts_voices import SUPPORTED_LANGUAGES
from inference.infer_tool import Svc
from utils import mix_model

import os
os.environ["LANG"] = "en_US.UTF-8"
os.environ["LC_ALL"] = "en_US.UTF-8"

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('markdown_it').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('multipart').setLevel(logging.WARNING)

model = None
spk = None
debug = False

local_model_root = './trained'

cuda = {}
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_properties(i).name
        cuda[f"CUDA:{i} {device_name}"] = f"cuda:{i}"

def upload_mix_append_file(files,sfiles):
    try:
        if(sfiles is None):
            file_paths = [file.name for file in files]
        else:
            file_paths = [file.name for file in chain(files,sfiles)]
        p = {file:100 for file in file_paths}
        return file_paths,mix_model_output1.update(value=json.dumps(p,indent=2))
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)

def mix_submit_click(js,mode):
    try:
        assert js.lstrip()!=""
        modes = {"convex_combination":0, "linear_combination":1}  # キーを英語化
        mode = modes[mode]
        data = json.loads(js)
        data = list(data.items())
        model_path,mix_rate = zip(*data)
        path = mix_model(model_path,mix_rate,mode)
        return f"成功しました。ファイルは{path}に保存されました"
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)

def updata_mix_info(files):
    try:
        if files is None :
            return mix_model_output1.update(value="")
        p = {file.name:100 for file in files}
        return mix_model_output1.update(value=json.dumps(p,indent=2))
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)

def modelAnalysis(model_path,config_path,cluster_model_path,device,enhance,diff_model_path,diff_config_path,only_diffusion,use_spk_mix,local_model_enabled,local_model_selection):
    global model
    try:
        device = cuda[device] if "CUDA" in device else device
        cluster_filepath = os.path.split(cluster_model_path.name) if cluster_model_path is not None else "no_cluster"
        # get model and config path
        if (local_model_enabled):
            # local path
            model_path = glob.glob(os.path.join(local_model_selection, '*.pth'))[0]
            config_path = glob.glob(os.path.join(local_model_selection, '*.json'))[0]
        else:
            # upload from webpage
            model_path = model_path.name
            config_path = config_path.name
        fr = ".pkl" in cluster_filepath[1]
        model = Svc(model_path,
                config_path,
                device=device if device != "Auto" else None,
                cluster_model_path = cluster_model_path.name if cluster_model_path is not None else "",
                nsf_hifigan_enhance=enhance,
                diffusion_model_path = diff_model_path.name if diff_model_path is not None else "",
                diffusion_config_path = diff_config_path.name if diff_config_path is not None else "",
                shallow_diffusion = True if diff_model_path is not None else False,
                only_diffusion = only_diffusion,
                spk_mix_enable = use_spk_mix,
                feature_retrieval = fr
                )
                
        spks = list(model.spk2id.keys())
        device_name = torch.cuda.get_device_properties(model.dev).name if "cuda" in str(model.dev) else str(model.dev)
        msg = f"モデルをデバイス{device_name}に正常にロードしました\n"
        if cluster_model_path is None:
            msg += "クラスタリングモデルまたは特徴検索モデルはロードされていません\n"
        elif fr:
            msg += f"特徴検索モデル{cluster_filepath[1]}を正常にロードしました\n"
        else:
            msg += f"クラスタリングモデル{cluster_filepath[1]}を正常にロードしました\n"
        if diff_model_path is None:
            msg += "拡散モデルはロードされていません\n"
        else:
            msg += f"拡散モデル{diff_model_path.name}を正常にロードしました\n"

        if len(spks) == 0:
            # 話者埋め込みなしのモデル
            msg += "このモデルは単一話者モデルです（話者選択なし）\n"
            return sid.update(choices = ["default"], value="default"), msg
        else:
            msg += "現在のモデルで利用可能な音色：\n"
            for i in spks:
                msg += i + " "
            return sid.update(choices = spks, value=spks[0]), msg

    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)

    
def modelUnload():
    global model
    if model is None:
        return sid.update(choices = [],value=""),"アンロードするモデルがありません！"
    else:
        model.unload_model()
        model = None
        torch.cuda.empty_cache()
        return sid.update(choices = [],value=""),"モデルのアンロードが完了しました！"
    
def vc_infer(output_format, sid, audio_path, truncated_basename, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment):
    global model
    _audio = model.slice_inference(
        audio_path,
        sid,
        vc_transform,
        slice_db,
        cluster_ratio,
        auto_f0,
        noise_scale,
        pad_seconds,
        cl_num,
        lg_num,
        lgr_num,
        f0_predictor,
        enhancer_adaptive_key,
        cr_threshold,
        k_step,
        use_spk_mix,
        second_encoding,
        loudness_envelope_adjustment
    )  
    model.clear_empty()
    # resultsフォルダ内に保存ファイルのパスを構築
    str(int(time.time()))
    if not os.path.exists("results"):
        os.makedirs("results")
    key = "auto" if auto_f0 else f"{int(vc_transform)}key"
    cluster = "_" if cluster_ratio == 0 else f"_{cluster_ratio}_"
    isdiffusion = "sovits"
    if model.shallow_diffusion:
        isdiffusion = "sovdiff"

    if model.only_diffusion:
        isdiffusion = "diff"
    
    output_file_name = 'result_'+truncated_basename+f'_{sid}_{key}{cluster}{isdiffusion}.{output_format}'
    output_file = os.path.join("results", output_file_name)
    soundfile.write(output_file, _audio, model.target_sample, format=output_format)
    return output_file

def vc_fn(sid, input_audio, output_format, vc_transform, auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,f0_predictor,enhancer_adaptive_key,cr_threshold,k_step,use_spk_mix,second_encoding,loudness_envelope_adjustment):
    global model
    try:
        if input_audio is None:
            return "オーディオをアップロードする必要があります", None
        if model is None:
            return "モデルをアップロードする必要があります", None
        if getattr(model, 'cluster_model', None) is None and model.feature_retrieval is False:
            if cluster_ratio != 0:
                return "クラスター比率を割り当てる前に、クラスタリングモデルまたは特徴検索モデルをアップロードする必要があります！", None
        #print(input_audio)    
        audio, sampling_rate = soundfile.read(input_audio)
        #print(audio.shape,sampling_rate)
        if np.issubdtype(audio.dtype, np.integer):
            audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        #print(audio.dtype)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        # 不明な理由でGradioアップロードのfilepathに奇妙な固定サフィックスがあるため、ここで削除
        truncated_basename = Path(input_audio).stem[:-6]
        processed_audio = os.path.join("raw", f"{truncated_basename}.wav")
        soundfile.write(processed_audio, audio, sampling_rate, format="wav")
        output_file = vc_infer(output_format, sid, processed_audio, truncated_basename, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment)

        return "成功", output_file
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)

def text_clear(text):
    return re.sub(r"[\n\,\(\) ]", "", text)

def vc_fn2(_text, _lang, _gender, _rate, _volume, sid, output_format, vc_transform, auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,f0_predictor,enhancer_adaptive_key,cr_threshold, k_step,use_spk_mix,second_encoding,loudness_envelope_adjustment):
    global model
    try:
        if model is None:
            return "モデルをアップロードする必要があります", None
        if getattr(model, 'cluster_model', None) is None and model.feature_retrieval is False:
            if cluster_ratio != 0:
                return "クラスター比率を割り当てる前に、クラスタリングモデルまたは特徴検索モデルをアップロードする必要があります！", None
        _rate = f"+{int(_rate*100)}%" if _rate >= 0 else f"{int(_rate*100)}%"
        _volume = f"+{int(_volume*100)}%" if _volume >= 0 else f"{int(_volume*100)}%"
        if _lang == "Auto":
            _gender = "Male" if _gender == "男性" else "Female"  # 日本語化
            subprocess.run([sys.executable, "edgetts/tts.py", _text, _lang, _rate, _volume, _gender])
        else:
            subprocess.run([sys.executable, "edgetts/tts.py", _text, _lang, _rate, _volume])
        target_sr = 44100
        y, sr = librosa.load("tts.wav")
        resampled_y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        soundfile.write("tts.wav", resampled_y, target_sr, subtype = "PCM_16")
        input_audio = "tts.wav"
        #audio, _ = soundfile.read(input_audio)
        output_file_path = vc_infer(output_format, sid, input_audio, "tts", vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment)
        os.remove("tts.wav")
        return "成功", output_file_path
    except Exception as e:
        if debug: traceback.print_exc()  # noqa: E701
        raise gr.Error(e)

def model_compression(_model):
    if _model == "":
        return "まず圧縮するモデルを選択してください"
    else:
        model_path = os.path.split(_model.name)
        filename, extension = os.path.splitext(model_path[1])
        output_model_name = f"{filename}_compressed{extension}"
        output_path = os.path.join(os.getcwd(), output_model_name)
        removeOptimizer(_model.name, output_path)
        return f"モデルは{output_path}に正常に保存されました"

def scan_local_models():
    res = []
    candidates = glob.glob(os.path.join(local_model_root, '**', '*.json'), recursive=True)
    candidates = set([os.path.dirname(c) for c in candidates])
    for candidate in candidates:
        jsons = glob.glob(os.path.join(candidate, '*.json'))
        pths = glob.glob(os.path.join(candidate, '*.pth'))
        if (len(jsons) == 1 and len(pths) == 1):
            # must contain exactly one json and one pth file
            res.append(candidate)
    return res

def local_model_refresh_fn():
    choices = scan_local_models()
    return gr.Dropdown.update(choices=choices)

def debug_change():
    global debug
    debug = debug_button.value

with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue = gr.themes.colors.green,
        font=["Source Sans Pro", "Arial", "sans-serif"],
        font_mono=['JetBrains mono', "Consolas", 'Courier New']
    ),
) as app:
    with gr.Tabs():
        with gr.TabItem("推論"):
            gr.Markdown(value="""
                So-vits-svc 4.0 推論 webui
                """)
            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(value="""
                        <font size=2> モデル設定</font>
                        """)
                    with gr.Tabs():
                        # invisible checkbox that tracks tab status
                        local_model_enabled = gr.Checkbox(value=False, visible=False)
                        with gr.TabItem('アップロード') as local_model_tab_upload:
                            with gr.Row():
                                model_path = gr.File(label="モデルファイルを選択")
                                config_path = gr.File(label="設定ファイルを選択")
                        with gr.TabItem('ローカル') as local_model_tab_local:
                            gr.Markdown(f'モデルは{local_model_root}フォルダ内に配置する必要があります')
                            local_model_refresh_btn = gr.Button('ローカルモデルリストを更新')
                            local_model_selection = gr.Dropdown(label='モデルフォルダを選択', choices=[], interactive=True)
                    with gr.Row():
                        diff_model_path = gr.File(label="拡散モデルファイルを選択")
                        diff_config_path = gr.File(label="拡散モデル設定ファイルを選択")
                    cluster_model_path = gr.File(label="クラスタリングモデルまたは特徴検索ファイルを選択（なければ選択不要）")
                    device = gr.Dropdown(label="推論デバイス、デフォルトはCPUとGPUの自動選択", choices=["Auto",*cuda.keys(),"cpu"], value="Auto")
                    enhance = gr.Checkbox(label="NSF_HIFIGAN強化を使用するかどうか。このオプションは、一部の訓練セットが少ないモデルに対して一定の音質向上効果がありますが、訓練済みモデルには逆効果があります。デフォルトはオフ", value=False)
                    only_diffusion = gr.Checkbox(label="完全拡散推論を使用するかどうか。有効にすると、So-VITSモデルを使用せず、拡散モデルのみを使用して完全な拡散推論を行います。デフォルトはオフ", value=False)
                with gr.Column():
                    gr.Markdown(value="""
                        <font size=3>左側のファイルをすべて選択した後（すべてのファイルモジュールにdownloadと表示される）、「モデルをロード」をクリックして解析します：</font>
                        """)
                    model_load_button = gr.Button(value="モデルをロード", variant="primary")
                    model_unload_button = gr.Button(value="モデルをアンロード", variant="primary")
                    sid = gr.Dropdown(label="音色（話者）")
                    sid_output = gr.Textbox(label="出力メッセージ")


            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(value="""
                        <font size=2> 推論設定</font>
                        """)
                    auto_f0 = gr.Checkbox(label="自動f0予測。クラスタリングモデルと組み合わせるとf0予測効果が向上しますが、変調機能が無効になります（音声変換のみ。歌声でチェックすると極度に音痴になります）", value=False)
                    f0_predictor = gr.Dropdown(label="F0予測器を選択。crepe、pm、dio、harvest、rmvpeから選択可能。デフォルトはpm（注意：crepeは元のF0に平均フィルタを使用）", choices=["pm","dio","harvest","crepe","rmvpe"], value="pm")
                    vc_transform = gr.Number(label="変調（整数、正負可、半音数、1オクターブ上げる場合は12）", value=0)
                    cluster_ratio = gr.Number(label="クラスタリングモデル/特徴検索混合比率、0-1の間、0はクラスタリング/特徴検索を有効にしません。クラスタリング/特徴検索を使用すると音色の類似度が向上しますが、発音が低下します（使用する場合は0.5程度を推奨）", value=0)
                    slice_db = gr.Number(label="スライス閾値", value=-40)
                    output_format = gr.Radio(label="オーディオ出力フォーマット", choices=["wav", "flac", "mp3"], value = "wav")
                    noise_scale = gr.Number(label="noise_scale 変更非推奨、音質に影響します。玄学パラメータ", value=0.4)
                    k_step = gr.Slider(label="浅い拡散ステップ数。拡散モデルを使用した場合のみ有効。ステップ数が多いほど拡散モデルの結果に近づきます", value=100, minimum = 1, maximum = 1000)
                with gr.Column():
                    pad_seconds = gr.Number(label="推論オーディオpadの秒数。不明な理由で冒頭と末尾にノイズが発生するため、短い無音セグメントをpadすると発生しません", value=0.5)
                    cl_num = gr.Number(label="オーディオ自動スライス、0はスライスなし、単位は秒(s)", value=0)
                    lg_num = gr.Number(label="両端のオーディオスライスのクロスフェードイン長さ。自動スライス後に音声が不連続な場合、この値を調整できます。連続している場合はデフォルト値0の使用を推奨。この設定は推論速度に影響します。単位は秒/s", value=0)
                    lgr_num = gr.Number(label="自動オーディオスライス後、各スライスの先頭と末尾を破棄する必要があります。このパラメータはクロス長さを保持する比率を設定します。範囲0-1、左開右閉", value=0.75)
                    enhancer_adaptive_key = gr.Number(label="エンハンサーをより高い音域に適応させる（単位は半音数）|デフォルトは0", value=0)
                    cr_threshold = gr.Number(label="F0フィルタリング閾値、crepe起動時のみ有効。数値範囲は0-1。この値を下げると音痴の確率が減りますが、無音が増えます", value=0.05)
                    loudness_envelope_adjustment = gr.Number(label="入力ソースのラウドネスエンベロープを出力ラウドネスエンベロープに置き換える融合比率、1に近いほど出力ラウドネスエンベロープを使用", value = 0)
                    second_encoding = gr.Checkbox(label = "二次エンコーディング。浅い拡散前に元のオーディオを二次エンコードします。玄学オプション、効果は時々良く時々悪い。デフォルトはオフ", value=False)
                    use_spk_mix = gr.Checkbox(label = "動的声線融合", value = False, interactive = False)
            with gr.Tabs():
                with gr.TabItem("オーディオ→オーディオ"):
                    vc_input3 = gr.Audio(label="オーディオを選択", type="filepath")
                    vc_submit = gr.Button("オーディオ変換", variant="primary")
                with gr.TabItem("テキスト→オーディオ"):
                    text2tts=gr.Textbox(label="変換するテキストをここに入力します。注意：この機能を使用する場合、F0予測を有効にすることをお勧めします。そうしないと奇妙になります")
                    with gr.Row():
                        tts_gender = gr.Radio(label = "話者の性別", choices = ["男性","女性"], value = "男性")
                        tts_lang = gr.Dropdown(label = "言語を選択、Autoは入力テキストから自動認識", choices=SUPPORTED_LANGUAGES, value = "Auto")
                        tts_rate = gr.Slider(label = "TTS音声変速（倍速相対値）", minimum = -1, maximum = 3, value = 0, step = 0.1)
                        tts_volume = gr.Slider(label = "TTS音声ボリューム（相対値）", minimum = -1, maximum = 1.5, value = 0, step = 0.1)
                    vc_submit2 = gr.Button("テキスト変換", variant="primary")
            with gr.Row():
                with gr.Column():
                    vc_output1 = gr.Textbox(label="出力メッセージ")
                with gr.Column():
                    vc_output2 = gr.Audio(label="出力オーディオ", interactive=False)

        with gr.TabItem("小ツール/実験的機能"):
            gr.Markdown(value="""
                        <font size=2> So-vits-svc 4.0 小ツール/実験的機能</font>
                        """)
            with gr.Tabs():
                with gr.TabItem("静的声線融合"):
                    gr.Markdown(value="""
                        <font size=2> 紹介：この機能は複数の音声モデルを1つの音声モデルに合成できます（複数のモデルパラメータの凸結合または線形結合）。現実には存在しない声を作り出します
                                          注意：
                                          1.この機能は単一話者モデルのみをサポートします
                                          2.複数話者モデルを強制的に使用する場合、複数のモデルの話者数が同じであることを確認する必要があります。これにより、同じSpeakerID下の音声を混合できます
                                          3.すべての混合対象モデルのconfig.jsonのmodelフィールドが同じであることを確認してください
                                          4.出力された混合モデルは、合成対象モデルの任意のconfig.jsonを使用できますが、クラスタリングモデルは使用できません
                                          5.モデルを一括アップロードする場合、モデルをフォルダに入れてから一緒にアップロードするのが最善です
                                          6.混合比率の調整は0-100の間を推奨しますが、他の数値にも調整できます。ただし、線形結合モードでは未知の効果が発生します
                                          7.混合完了後、ファイルはプロジェクトのルートディレクトリに保存され、ファイル名はoutput.pthになります
                                          8.凸結合モードは混合比率にSoftmaxを実行して混合比率の合計を1にしますが、線形結合モードは実行しません
                        </font>
                        """)
                    mix_model_path = gr.Files(label="混合するモデルファイルを選択")
                    mix_model_upload_button = gr.UploadButton("混合するモデルファイルを選択/追加", file_count="multiple")
                    mix_model_output1 = gr.Textbox(
                                            label="混合比率の調整、単位/%",
                                            interactive = True
                                         )
                    mix_mode = gr.Radio(choices=["凸結合", "線形結合"], label="融合モード",value="凸結合",interactive = True)
                    mix_submit = gr.Button("声線融合開始", variant="primary")
                    mix_model_output2 = gr.Textbox(
                                            label="出力メッセージ"
                                         )
                    mix_model_path.change(updata_mix_info,[mix_model_path],[mix_model_output1])
                    mix_model_upload_button.upload(upload_mix_append_file, [mix_model_upload_button,mix_model_path], [mix_model_path,mix_model_output1])
                    mix_submit.click(mix_submit_click, [mix_model_output1,mix_mode], [mix_model_output2])
                
                with gr.TabItem("モデル圧縮ツール"):
                    gr.Markdown(value="""
                        このツールはモデルのサイズ圧縮を実現できます。**モデルの推論機能に影響を与えることなく**、元々約600MのSo-VITSモデルを約200Mに圧縮し、ハードディスクの負担を大幅に軽減します。
                        **注意：圧縮後のモデルは訓練を継続できません。確実に完成した後に圧縮してください。**
                    """)
                    model_to_compress = gr.File(label="モデルアップロード")
                    compress_model_btn = gr.Button("モデルを圧縮", variant="primary")
                    compress_model_output = gr.Textbox(label="出力情報", value="")

                    compress_model_btn.click(model_compression, [model_to_compress], [compress_model_output])
                    
                    
    with gr.Tabs():
        with gr.Row(variant="panel"):
            with gr.Column():
                gr.Markdown(value="""
                    <font size=2> WebUI設定</font>
                    """)
                debug_button = gr.Checkbox(label="デバッグモード。コミュニティにバグを報告する必要がある場合は有効にしてください。有効にすると、コンソールに具体的なエラーメッセージが表示されます", value=debug)
        # refresh local model list
        local_model_refresh_btn.click(local_model_refresh_fn, outputs=local_model_selection)
        # set local enabled/disabled on tab switch
        local_model_tab_upload.select(lambda: False, outputs=local_model_enabled)
        local_model_tab_local.select(lambda: True, outputs=local_model_enabled)
        
        vc_submit.click(vc_fn, [sid, vc_input3, output_format, vc_transform,auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,f0_predictor,enhancer_adaptive_key,cr_threshold,k_step,use_spk_mix,second_encoding,loudness_envelope_adjustment], [vc_output1, vc_output2])
        vc_submit2.click(vc_fn2, [text2tts, tts_lang, tts_gender, tts_rate, tts_volume, sid, output_format, vc_transform,auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,f0_predictor,enhancer_adaptive_key,cr_threshold,k_step,use_spk_mix,second_encoding,loudness_envelope_adjustment], [vc_output1, vc_output2])

        debug_button.change(debug_change,[],[])
        model_load_button.click(modelAnalysis,[model_path,config_path,cluster_model_path,device,enhance,diff_model_path,diff_config_path,only_diffusion,use_spk_mix,local_model_enabled,local_model_selection],[sid,sid_output])
        model_unload_button.click(modelUnload,[],[sid,sid_output])
    os.system("start http://127.0.0.1:7860")
    app.launch()
    