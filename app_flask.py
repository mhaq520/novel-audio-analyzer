import os
import time
import json
import threading
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
from core.cache_manager import get_cached_transcript, save_cached_transcript
from core.asr_engine import transcribe_audio, release_asr_model
from core.llm_engine import analyze_text, release_llm

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB 上限

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('./output', exist_ok=True)

# 全局变量存储任务状态
task_status = {}
task_lock = threading.Lock()


def process_files_async(file_paths, task_id):
    """后台处理任务"""
    global task_status
    try:
        with task_lock:
            task_status[task_id] = {'progress': 0, 'logs': [], 'status': 'running'}

        logs = []
        total = len(file_paths)

        # 第一阶段：ASR
        logs.append("准备 ASR 模型...")
        with task_lock:
            task_status[task_id]['logs'] = logs
            task_status[task_id]['progress'] = 5

        need_transcribe = []
        for path in file_paths:
            cached = get_cached_transcript(path)
            if cached is None:
                need_transcribe.append(path)

        if need_transcribe:
            logs.append(f"ASR 转写中（共 {len(need_transcribe)} 个文件）...")
            with task_lock:
                task_status[task_id]['logs'] = logs
            for idx, path in enumerate(need_transcribe):
                logs.append(f"转写: {os.path.basename(path)}")
                with task_lock:
                    task_status[task_id]['logs'] = logs
                    task_status[task_id]['progress'] = 10 + int(30 * idx / len(need_transcribe))
                try:
                    transcript = transcribe_audio(path)
                    save_cached_transcript(path, transcript)
                    logs.append(f"✅ 转写完成: {os.path.basename(path)}")
                except Exception as e:
                    logs.append(f"❌ 转写失败 {os.path.basename(path)}: {str(e)}")
            release_asr_model()
        else:
            logs.append("✅ 所有文件已有缓存，跳过 ASR。")

        # 第二阶段：LLM 分析
        logs.append("加载 LLM 模型...")
        with task_lock:
            task_status[task_id]['logs'] = logs
            task_status[task_id]['progress'] = 60

        for idx, path in enumerate(file_paths):
            base_name = os.path.splitext(os.path.basename(path))[0]
            output_txt = os.path.join('./output', f"{base_name}_分析.txt")

            if os.path.exists(output_txt):
                logs.append(f"⏩ 已存在: {base_name}")
                with task_lock:
                    task_status[task_id]['logs'] = logs
                continue

            transcript = get_cached_transcript(path)
            if transcript is None:
                logs.append(f"⚠️ 无转写缓存: {base_name}")
                with task_lock:
                    task_status[task_id]['logs'] = logs
                continue

            logs.append(f"分析: {base_name}")
            with task_lock:
                task_status[task_id]['logs'] = logs
                task_status[task_id]['progress'] = 60 + int(35 * (idx + 1) / total)
            try:
                summary, keywords = analyze_text(transcript)
                with open(output_txt, 'w', encoding='utf-8') as f:
                    f.write(f"【剧情摘要】\n{summary}\n\n【行为关键词】\n{', '.join(keywords)}")
                logs.append(f"✅ 分析完成: {base_name}")
            except Exception as e:
                logs.append(f"❌ 分析失败 {base_name}: {str(e)}")

        release_llm()
        logs.append("🏁 所有任务处理完毕！")
        with task_lock:
            task_status[task_id]['logs'] = logs
            task_status[task_id]['progress'] = 100
            task_status[task_id]['status'] = 'completed'

    except Exception as e:
        logs.append(f"❌ 全局错误: {str(e)}")
        with task_lock:
            task_status[task_id]['logs'] = logs
            task_status[task_id]['status'] = 'failed'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    uploaded_files = request.files.getlist('files')
    if not uploaded_files or uploaded_files[0].filename == '':
        return jsonify({'error': 'Empty file list'}), 400

    # 保存上传的文件
    saved_paths = []
    for file in uploaded_files:
        filename = os.path.basename(file.filename)  # 获取文件名，去除目录部分
        filename = filename.replace('/', '_').replace('\\', '_')  # 替换危险字符
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        saved_paths.append(save_path)

    # 生成任务ID
    task_id = str(int(time.time() * 1000))

    # 启动后台线程处理
    thread = threading.Thread(target=process_files_async, args=(saved_paths, task_id))
    thread.daemon = True
    thread.start()

    return jsonify({'task_id': task_id})


@app.route('/status/<task_id>')
def get_status(task_id):
    with task_lock:
        status = task_status.get(task_id)
    if not status:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(status)


@app.route('/output/<filename>')
def get_output(filename):
    return send_from_directory('./output', filename)


@app.route('/list_outputs')
def list_outputs():
    files = os.listdir('./output')
    return jsonify([f for f in files if f.endswith('.txt')])


@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join('./output', filename), as_attachment=True)

@app.route('/open_folder', methods=['POST'])
def open_folder():
    import subprocess
    subprocess.Popen(['explorer', os.path.abspath('./output')])
    return '', 204


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)