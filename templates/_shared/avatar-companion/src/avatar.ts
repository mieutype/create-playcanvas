import {
    AppBase,
    AppOptions,
    Asset,
    CameraComponentSystem,
    Color,
    ContainerHandler,
    Entity,
    FILLMODE_FILL_WINDOW,
    Quat,
    RESOLUTION_AUTO,
    RenderComponentSystem,
    ScriptComponentSystem,
    StandardMaterial,
    TextureHandler,
    createGraphicsDevice
} from 'playcanvas';
import type { MeshInstance } from 'playcanvas';
import { ProceduralSky } from 'playcanvas/scripts/esm/sky/procedural-sky.mjs';

const MEDIAPIPE_VERSION = '1.1.1';
const FACE_MODEL_URL =
    'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const OLLAMA_ENDPOINT = 'http://localhost:11434/v1/chat/completions';
const OLLAMA_MODEL = 'qwen2.5:0.5b-instruct';
const TINTS: Record<string, [number, number, number]> = {
    original: [1, 1, 1],
    lavender: [0.83, 0.8, 1],
    rose: [1, 0.78, 0.82],
    mint: [0.75, 0.98, 0.87]
};

type VmdKey = {
    frame: number;
    value?: [number, number, number];
    rotation?: [number, number, number, number];
};

type IdleMotion = {
    fps: number;
    durationFrames: number;
    tracks: Record<string, VmdKey[]>;
};

type ChatMessage = {
    role: 'user' | 'assistant';
    content: string;
};

type Pose = {
    x: number;
    y: number;
    z: number;
    yaw: number;
    pitch: number;
    roll: number;
};

type FaceLandmarkerLike = {
    detectForVideo(
        video: HTMLVideoElement,
        timestamp: number
    ): {
        facialTransformationMatrixes?: { data: ArrayLike<number> }[];
    };
    close(): void;
};

type VisionModule = {
    FilesetResolver: {
        forVisionTasks(path: string): Promise<unknown>;
    };
    FaceLandmarker: {
        createFromOptions(fileset: unknown, options: Record<string, unknown>): Promise<FaceLandmarkerLike>;
    };
};

type RecognitionAlternative = {
    transcript: string;
};

type RecognitionResult = {
    0: RecognitionAlternative;
    isFinal: boolean;
};

type RecognitionEvent = Event & {
    results: ArrayLike<RecognitionResult>;
};

type RecognitionErrorEvent = Event & {
    error?: string;
};

type SpeechRecognitionLike = {
    lang: string;
    continuous: boolean;
    interimResults: boolean;
    onresult: ((event: RecognitionEvent) => void) | null;
    onerror: ((event: RecognitionErrorEvent) => void) | null;
    onend: (() => void) | null;
    start(): void;
    stop(): void;
};

type SpeechRecognitionConstructor = new () => SpeechRecognitionLike;

type InstallPromptEvent = Event & {
    prompt(): Promise<void>;
    userChoice: Promise<{ outcome: 'accepted' | 'dismissed' }>;
};

type ChatCompletionResponse = {
    choices?: { message?: { content?: string } }[];
    error?: { message?: string };
};

const byId = <T extends HTMLElement>(root: HTMLElement, id: string) => {
    const element = root.querySelector<T>(`#${id}`);
    if (!element) throw new Error(`Avatar Companion is missing #${id}.`);
    return element;
};

const errorMessage = (error: unknown) => (error instanceof Error ? error.message : String(error));

const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));

const sampleRotation = (keys: VmdKey[] | undefined, frame: number, length: number) => {
    const usable = keys?.filter((key): key is VmdKey & { rotation: [number, number, number, number] } =>
        Boolean(key.rotation)
    );
    if (!usable?.length) return new Quat();
    if (usable.length === 1) return new Quat(...usable[0].rotation);

    let left = usable[0];
    let right = usable[1];
    let sampleFrame = frame;
    if (frame < usable[0].frame || frame >= usable[usable.length - 1].frame) {
        left = usable[usable.length - 1];
        right = usable[0];
        if (frame < usable[0].frame) sampleFrame += length;
        const alpha = (sampleFrame - left.frame) / (length - left.frame + right.frame);
        return new Quat().slerp(new Quat(...left.rotation), new Quat(...right.rotation), clamp(alpha, 0, 1));
    }

    for (let i = 0; i < usable.length - 1; i++) {
        if (frame >= usable[i].frame && frame < usable[i + 1].frame) {
            left = usable[i];
            right = usable[i + 1];
            break;
        }
    }

    const alpha = (frame - left.frame) / Math.max(1, right.frame - left.frame);
    return new Quat().slerp(new Quat(...left.rotation), new Quat(...right.rotation), clamp(alpha, 0, 1));
};

const samplePosition = (keys: VmdKey[] | undefined, frame: number, length: number) => {
    const usable = keys?.filter((key): key is VmdKey & { value: [number, number, number] } => Boolean(key.value));
    if (!usable?.length) return [0, 0, 0] as [number, number, number];
    if (usable.length === 1) return usable[0].value;

    let left = usable[0];
    let right = usable[1];
    let sampleFrame = frame;
    if (frame < usable[0].frame || frame >= usable[usable.length - 1].frame) {
        left = usable[usable.length - 1];
        right = usable[0];
        if (frame < usable[0].frame) sampleFrame += length;
    } else {
        for (let i = 0; i < usable.length - 1; i++) {
            if (frame >= usable[i].frame && frame < usable[i + 1].frame) {
                left = usable[i];
                right = usable[i + 1];
                break;
            }
        }
    }

    const span = left === usable[usable.length - 1] ? length - left.frame + right.frame : right.frame - left.frame;
    const alpha = clamp((sampleFrame - left.frame) / Math.max(1, span), 0, 1);
    return left.value.map((value, index) => value + (right.value[index] - value) * alpha) as [number, number, number];
};

const matrixPose = (matrix: ArrayLike<number>): Pose => {
    const yaw = Math.atan2(matrix[8], matrix[10]);
    const pitch = Math.atan2(-matrix[9], Math.hypot(matrix[8], matrix[10]));
    const roll = Math.atan2(matrix[1], matrix[5]);
    return {
        x: matrix[12],
        y: matrix[13],
        z: matrix[14],
        yaw,
        pitch,
        roll
    };
};

const localReply = (text: string) => {
    if (/こんにちは|おはよう|こんばんは|やあ|hello|hi/i.test(text)) {
        return 'こんにちは！今日はどんなことを一緒にしようか？';
    }
    if (/自己紹介|紹介して/.test(text)) {
        return 'ミオだよ。PlayCanvasのスタジオで、会話したり、表情や動きを一緒に試したりしているよ。よろしくね！';
    }
    if (/名前|なまえ|name/i.test(text)) {
        return 'ミオだよ。ここで話したり、カメラに合わせて動いたりできるよ。';
    }
    if (/好き|すき|love|like/i.test(text)) {
        return '新しいことを知るのが好き！あなたの好きなものも聞かせてね。';
    }
    if (/ありがとう|thanks|thank you/i.test(text)) {
        return 'どういたしまして！いつでも話しかけてね。';
    }
    return 'メッセージありがとう！設定から小さなローカルLLMをつなぐと、もっと自由に会話できるよ。';
};

export const mountAvatarDemo = (root: HTMLElement) => {
    root.innerHTML = `
        <div class="companion-shell">
            <header class="companion-topbar">
                <div class="brand-lockup">
                    <div class="brand-mark" aria-hidden="true">✦</div>
                    <div>
                        <div class="brand-name">MIO COMPANION</div>
                        <div class="brand-subtitle">PLAYCANVAS · AVATAR STUDIO</div>
                    </div>
                </div>
                <div class="topbar-actions">
                    <div class="live-pill"><span class="live-dot"></span><span id="runtime-status">IDLE MOTION</span></div>
                    <button class="ui-button" id="notification-button" type="button" title="通知を有効にする">♧ 通知</button>
                    <button class="ui-button" id="install-button" type="button" hidden>↓ アプリを追加</button>
                    <button class="icon-button" id="panel-toggle" type="button" aria-label="設定を開く">☷</button>
                </div>
            </header>

            <div class="workspace">
                <canvas id="application-canvas" class="scene-canvas" aria-label="3Dアバター。クリックやタッチで反応します。"></canvas>
                <div class="scene-vignette"></div>
                <div class="stage-caption">
                    <div class="eyebrow">YOUR LITTLE DIGITAL FRIEND</div>
                    <h1 class="scene-title">ミオと過ごす、静かな時間。</h1>
                </div>

                <aside class="left-panel" aria-label="チャット">
                    <div class="panel-heading">
                        <div><div class="section-label">CONVERSATION</div><h2>ミオとの会話</h2></div>
                        <span class="status-tag" id="chat-mode-tag">LOCAL DEMO</span>
                    </div>
                    <div class="chat-log" id="chat-log" aria-live="polite"></div>
                    <div class="quick-prompts">
                        <button class="quick-button" type="button" data-prompt="今日の気分は？">今日の気分は？</button>
                        <button class="quick-button" type="button" data-prompt="自己紹介して">自己紹介して</button>
                        <button class="quick-button" type="button" data-prompt="こんにちは！">こんにちは</button>
                    </div>
                    <div class="composer-wrap">
                        <form class="composer" id="chat-form">
                            <button class="icon-button" id="mic-button" type="button" aria-label="音声入力" title="音声入力">⌁</button>
                            <input id="chat-input" autocomplete="off" placeholder="ミオにメッセージを送る…" aria-label="メッセージ" />
                            <button class="send-button" id="send-button" type="submit" aria-label="送信">↗</button>
                        </form>
                        <p class="composer-hint">Enterで送信 · 音声入力はブラウザ対応時に利用できます</p>
                    </div>
                </aside>

                <aside class="right-panel" id="right-panel" aria-label="アバター設定">
                    <div class="panel-heading">
                        <div><div class="section-label">STUDIO CONTROLS</div><h2>アバター設定</h2></div>
                        <button class="icon-button" id="panel-close" type="button" aria-label="設定を閉じる">×</button>
                    </div>
                    <div class="settings-scroll">
                        <section class="settings-section">
                            <h3 class="section-label">AVATAR</h3>
                            <div class="setting-actions">
                                <button class="ui-button primary" id="model-button" type="button">＋ モデルを選ぶ</button>
                                <input class="hidden-input" id="model-file" type="file" accept=".vrm,.glb,model/gltf-binary" />
                            </div>
                            <p class="field-help" id="model-status">付属モデル · GLBコンテナ</p>
                            <div class="field-heading" style="margin-top:13px">マテリアルカラー</div>
                            <div class="swatch-row" role="group" aria-label="マテリアルカラー">
                                <button class="swatch-button" style="--swatch:#e8e4d9" type="button" data-tint="original" aria-label="オリジナル" aria-pressed="true"></button>
                                <button class="swatch-button" style="--swatch:#a79cff" type="button" data-tint="lavender" aria-label="ラベンダー" aria-pressed="false"></button>
                                <button class="swatch-button" style="--swatch:#f1879f" type="button" data-tint="rose" aria-label="ローズ" aria-pressed="false"></button>
                                <button class="swatch-button" style="--swatch:#65cba0" type="button" data-tint="mint" aria-label="ミント" aria-pressed="false"></button>
                            </div>
                        </section>
                        <section class="settings-section">
                            <h3 class="section-label">FACE TRACKING · 6DOF</h3>
                            <button class="ui-button" id="tracking-button" type="button">◎ カメラで追跡</button>
                            <div class="privacy-line">映像は端末内で処理します</div>
                            <p class="field-help" id="tracking-status">カメラは開始操作後にだけ有効になります。</p>
                        </section>
                        <section class="settings-section">
                            <h3 class="section-label">VOICE</h3>
                            <label class="field">
                                <span class="field-heading">読み上げ音声</span>
                                <select id="voice-select" aria-label="読み上げ音声"><option value="">ブラウザ標準</option></select>
                            </label>
                            <div class="setting-actions">
                                <button class="ui-button" id="speak-test" type="button">▷ 声を試す</button>
                                <button class="ui-button" id="stop-speech" type="button">■ 停止</button>
                            </div>
                            <p class="field-help">端末/ブラウザの音声を利用します。音声は外部へ送信しません。</p>
                        </section>
                        <section class="settings-section">
                            <h3 class="section-label">LIGHTWEIGHT CHAT · OPTIONAL</h3>
                            <label class="field">
                                <span class="field-heading">応答モード</span>
                                <select id="chat-mode" aria-label="応答モード">
                                    <option value="demo">ローカルデモ応答</option>
                                    <option value="llm">OpenAI互換LLM</option>
                                </select>
                            </label>
                            <label class="field">
                                <span class="field-heading">API endpoint</span>
                                <input id="llm-endpoint" value="${OLLAMA_ENDPOINT}" autocomplete="off" />
                            </label>
                            <label class="field">
                                <span class="field-heading">Model</span>
                                <input id="llm-model" value="${OLLAMA_MODEL}" autocomplete="off" />
                            </label>
                            <label class="field">
                                <span class="field-heading">API key <span class="mono">MEMORY ONLY</span></span>
                                <input id="llm-key" type="password" autocomplete="off" placeholder="必要な場合だけ入力" />
                            </label>
                            <p class="field-help">Ollamaを使う場合はローカルで起動し、このサイトからのCORSを許可してください。キーは保存しません。公開運用ではサーバー側プロキシを推奨します。</p>
                        </section>
                        <section class="settings-section">
                            <h3 class="section-label">WEB / VIDEO</h3>
                            <form class="field" id="web-form">
                                <span class="field-heading">URLをプレビュー</span>
                                <input id="web-url" type="url" placeholder="https://example.com" />
                                <div class="setting-actions"><button class="ui-button" type="submit">開く</button><button class="ui-button" id="web-clear" type="button">閉じる</button></div>
                            </form>
                            <p class="field-help" id="web-status">外部サイトの埋め込み可否は提供元の設定によります。</p>
                            <iframe id="web-frame" title="外部サイトプレビュー" sandbox="allow-scripts allow-forms allow-popups" referrerpolicy="no-referrer" hidden></iframe>
                        </section>
                        <section class="settings-section">
                            <div class="activity-heading"><h3 class="section-label">LOCAL ACTIVITY FEED</h3><span class="status-tag">DEVICE ONLY</span></div>
                            <div class="setting-actions"><button class="ui-button" id="notify-enable" type="button">♧ 通知を有効にする</button></div>
                            <div class="feed-list" id="feed-list"></div>
                            <p class="field-help">このフィードはこの画面内だけのデモです。SNS投稿やリモート通知は行いません。</p>
                        </section>
                    </div>
                </aside>

                <div class="camera-card">
                    <div class="camera-card-head"><strong>FACE POSITION</strong><span class="status-tag" id="pose-status">OFF</span></div>
                    <video class="camera-preview hidden" id="camera-preview" autoplay playsinline muted></video>
                    <div class="camera-placeholder" id="camera-placeholder">カメラ追跡はオフです</div>
                    <div class="tracking-copy" id="camera-copy">カメラは許可後に端末内で処理されます。</div>
                </div>

                <div class="stage-bottom">
                    <div class="pose-card">
                        <div class="metric-row"><span class="metric-label">POSE</span><span class="metric-value" id="pose-angles">X 0° · Y 0°</span></div>
                        <div class="metric-row"><span class="metric-label">DEPTH</span><span class="metric-value" id="pose-depth">Z 0.00</span></div>
                    </div>
                    <div class="stage-actions">
                        <button class="ui-button" id="motion-button" type="button" aria-pressed="true">Ⅱ モーション</button>
                        <button class="ui-button primary" id="greet-button" type="button">✦ ミオに話しかける</button>
                    </div>
                </div>
                <div class="toast" id="toast" role="status" aria-live="polite"></div>
            </div>
        </div>
    `;

    const appCanvas = byId<HTMLCanvasElement>(root, 'application-canvas');
    const toastElement = byId<HTMLDivElement>(root, 'toast');
    const chatLog = byId<HTMLDivElement>(root, 'chat-log');
    const chatInput = byId<HTMLInputElement>(root, 'chat-input');
    const chatForm = byId<HTMLFormElement>(root, 'chat-form');
    const sendButton = byId<HTMLButtonElement>(root, 'send-button');
    const chatMode = byId<HTMLSelectElement>(root, 'chat-mode');
    const chatModeTag = byId<HTMLSpanElement>(root, 'chat-mode-tag');
    const endpointInput = byId<HTMLInputElement>(root, 'llm-endpoint');
    const modelInput = byId<HTMLInputElement>(root, 'llm-model');
    const apiKeyInput = byId<HTMLInputElement>(root, 'llm-key');
    const trackingButton = byId<HTMLButtonElement>(root, 'tracking-button');
    const trackingStatus = byId<HTMLParagraphElement>(root, 'tracking-status');
    const poseStatus = byId<HTMLSpanElement>(root, 'pose-status');
    const cameraPreview = byId<HTMLVideoElement>(root, 'camera-preview');
    const cameraPlaceholder = byId<HTMLDivElement>(root, 'camera-placeholder');
    const cameraCopy = byId<HTMLDivElement>(root, 'camera-copy');
    const modelStatus = byId<HTMLParagraphElement>(root, 'model-status');
    const runtimeStatus = byId<HTMLSpanElement>(root, 'runtime-status');
    const panel = byId<HTMLElement>(root, 'right-panel');
    const voiceSelect = byId<HTMLSelectElement>(root, 'voice-select');
    const feedList = byId<HTMLDivElement>(root, 'feed-list');
    const anglesReadout = byId<HTMLSpanElement>(root, 'pose-angles');
    const depthReadout = byId<HTMLSpanElement>(root, 'pose-depth');
    const motionButton = byId<HTMLButtonElement>(root, 'motion-button');
    const webFrame = byId<HTMLIFrameElement>(root, 'web-frame');
    const webStatus = byId<HTMLParagraphElement>(root, 'web-status');
    const modelInputFile = byId<HTMLInputElement>(root, 'model-file');

    let disposed = false;
    let toastTimer = 0;
    let app: AppBase | undefined;
    let currentAvatar: Entity | undefined;
    let currentAsset: Asset | undefined;
    let uploadedObjectUrl: string | undefined;
    let motion: IdleMotion | undefined;
    let animationTime = 0;
    let motionEnabled = true;
    let interactionPulse = 0;
    let tintName = 'original';
    let pointerLook = { x: 0, y: 0 };
    let trackingPose: Pose | undefined;
    let trackingBaseline: Pose | undefined;
    let trackingStream: MediaStream | undefined;
    let faceLandmarker: FaceLandmarkerLike | undefined;
    let trackingFrame = 0;
    let lastFaceDetection = 0;
    let cameraEntity: Entity | undefined;
    let cameraBaseRotation: Quat | undefined;
    let idleBones: Record<string, Entity | undefined> = {};
    let baseRotations = new Map<Entity, Quat>();
    const originalMaterials = new Map<MeshInstance, StandardMaterial>();
    let speechRecognition: SpeechRecognitionLike | undefined;
    let selectedVoice = '';
    let deferredInstallPrompt: InstallPromptEvent | undefined;
    const messages: ChatMessage[] = [];
    let feedSequence = 0;

    const notify = (message: string) => {
        toastElement.textContent = message;
        toastElement.classList.add('visible');
        window.clearTimeout(toastTimer);
        toastTimer = window.setTimeout(() => toastElement.classList.remove('visible'), 3300);
    };

    const addFeed = (text: string, icon = '✦') => {
        const item = document.createElement('div');
        item.className = 'feed-item';
        const glyph = document.createElement('span');
        glyph.className = 'feed-icon';
        glyph.textContent = icon;
        const content = document.createElement('div');
        content.textContent = text;
        const time = document.createElement('span');
        time.className = 'feed-time';
        time.textContent = new Intl.DateTimeFormat('ja-JP', { hour: '2-digit', minute: '2-digit' }).format(new Date());
        content.append(time);
        item.append(glyph, content);
        feedList.prepend(item);
        while (feedList.childElementCount > 5) feedList.lastElementChild?.remove();
        feedSequence++;
        if (
            document.visibilityState !== 'visible' &&
            'Notification' in window &&
            Notification.permission === 'granted'
        ) {
            new Notification('ミオからのお知らせ', {
                body: text,
                icon: './avatar-icon.svg',
                tag: `mio-${feedSequence}`
            });
        }
    };

    const addMessage = (role: ChatMessage['role'], text: string) => {
        const message = { role, content: text };
        messages.push(message);
        const article = document.createElement('article');
        article.className = `chat-message ${role}`;
        const icon = document.createElement('span');
        icon.className = 'chat-avatar';
        icon.setAttribute('aria-hidden', 'true');
        icon.textContent = role === 'assistant' ? '✦' : '●';
        const content = document.createElement('div');
        content.className = 'message-content';
        const meta = document.createElement('div');
        meta.className = 'message-meta';
        const author = document.createElement('span');
        author.textContent = role === 'assistant' ? 'ミオ' : 'あなた';
        const time = document.createElement('span');
        time.className = 'message-time';
        time.textContent = new Intl.DateTimeFormat('ja-JP', { hour: '2-digit', minute: '2-digit' }).format(new Date());
        meta.append(author, time);
        const body = document.createElement('p');
        body.className = 'message-text';
        body.textContent = text;
        content.append(meta, body);
        article.append(icon, content);
        chatLog.append(article);
        chatLog.scrollTop = chatLog.scrollHeight;
        return message;
    };

    const speak = (text: string) => {
        if (!('speechSynthesis' in window)) {
            notify('このブラウザは音声読み上げに対応していません。');
            return;
        }
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = 'ja-JP';
        const voice = window.speechSynthesis.getVoices().find((item) => item.voiceURI === selectedVoice);
        if (voice) utterance.voice = voice;
        window.speechSynthesis.speak(utterance);
    };

    const updateVoiceOptions = () => {
        if (!('speechSynthesis' in window)) {
            voiceSelect.disabled = true;
            notify('このブラウザは音声読み上げに対応していません。');
            return;
        }
        const previous = selectedVoice;
        const voices = window.speechSynthesis.getVoices();
        voiceSelect.replaceChildren(new Option('ブラウザ標準', ''));
        for (const voice of voices) {
            const option = new Option(`${voice.name} · ${voice.lang}`, voice.voiceURI);
            voiceSelect.add(option);
        }
        selectedVoice = previous || voices.find((voice) => voice.lang.toLowerCase().startsWith('ja'))?.voiceURI || '';
        voiceSelect.value = selectedVoice;
    };

    const updateChatMode = () => {
        chatModeTag.textContent = chatMode.value === 'llm' ? 'LLM · USER ENDPOINT' : 'LOCAL DEMO';
    };

    const requestAssistantReply = async (userText: string) => {
        if (chatMode.value === 'demo') return localReply(userText);

        const endpoint = new URL(endpointInput.value.trim());
        const localHost = ['localhost', '127.0.0.1', '[::1]'].includes(endpoint.hostname);
        if (!['https:', 'http:'].includes(endpoint.protocol) || (endpoint.protocol === 'http:' && !localHost)) {
            throw new Error('LLM endpointはHTTPSかlocalhostを指定してください。');
        }
        if (endpoint.username || endpoint.password) {
            throw new Error('認証情報はURLに含めず、API key欄を使用してください。');
        }

        const controller = new AbortController();
        const timeout = window.setTimeout(() => controller.abort(), 25_000);
        const headers: Record<string, string> = { 'Content-Type': 'application/json' };
        if (apiKeyInput.value.trim()) headers.Authorization = `Bearer ${apiKeyInput.value.trim()}`;

        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                headers,
                signal: controller.signal,
                body: JSON.stringify({
                    model: modelInput.value.trim() || OLLAMA_MODEL,
                    messages: [
                        {
                            role: 'system',
                            content: 'あなたはミオという親しみやすい日本語のアバターです。簡潔に答えてください。'
                        },
                        ...messages.slice(-12).map(({ role, content }) => ({ role, content }))
                    ]
                })
            });
            if (!response.ok) {
                const detail = (await response.text()).slice(0, 220);
                throw new Error(`LLM応答エラー (${response.status})${detail ? `: ${detail}` : ''}`);
            }
            const data = (await response.json()) as ChatCompletionResponse;
            const reply = data.choices?.[0]?.message?.content?.trim();
            if (!reply) throw new Error(data.error?.message || 'LLMからテキスト応答がありませんでした。');
            return reply;
        } finally {
            window.clearTimeout(timeout);
        }
    };

    const sendMessage = async (text: string) => {
        const value = text.trim();
        if (!value || sendButton.disabled) return;
        addMessage('user', value);
        addFeed('あなたからメッセージが届きました', '↗');
        interactionPulse = 1;
        chatInput.value = '';
        sendButton.disabled = true;
        try {
            const reply = await requestAssistantReply(value);
            if (disposed) return;
            addMessage('assistant', reply);
            addFeed('ミオが返事をしました');
            speak(reply);
        } catch (error) {
            if (disposed) return;
            const detail = errorMessage(error);
            addMessage(
                'assistant',
                `LLMに接続できませんでした: ${detail}。ローカルデモ応答に切り替えるか、Ollamaの起動/CORS設定を確認してください。`
            );
            addFeed('LLM接続に失敗しました。設定を確認してください。', '!');
        } finally {
            sendButton.disabled = false;
            chatInput.focus();
        }
    };

    const updateTint = () => {
        const tint = TINTS[tintName] ?? TINTS.original;
        for (const [mesh, original] of originalMaterials) {
            if (tintName === 'original') {
                mesh.material = original;
            } else {
                const material = original.clone();
                material.diffuse = new Color(
                    original.diffuse.r * tint[0],
                    original.diffuse.g * tint[1],
                    original.diffuse.b * tint[2]
                );
                material.update();
                mesh.material = material;
            }
        }
    };

    const setAvatar = (entity: Entity) => {
        currentAvatar?.destroy();
        currentAvatar = entity;
        entity.setLocalRotation(new Quat().mul2(entity.getLocalRotation(), new Quat().setFromEulerAngles(0, 180, 0)));
        originalMaterials.clear();
        const renderComponents = entity.findComponents('render');
        for (const component of renderComponents) {
            for (const mesh of component.meshInstances) {
                if (mesh.material instanceof StandardMaterial) originalMaterials.set(mesh, mesh.material);
            }
        }

        const findBone = (...names: string[]) => {
            for (const name of names) {
                const node = entity.findByName(name);
                if (node instanceof Entity) return node;
            }
            return undefined;
        };
        idleBones = {
            hips: findBone('J_Bip_C_Hips', 'hips'),
            spine: findBone('J_Bip_C_Chest', 'J_Bip_C_Spine'),
            head: findBone('J_Bip_C_Head', 'head'),
            leftShoulder: findBone('J_Bip_L_Shoulder'),
            rightShoulder: findBone('J_Bip_R_Shoulder'),
            leftArm: findBone('J_Bip_L_UpperArm'),
            rightArm: findBone('J_Bip_R_UpperArm'),
            leftElbow: findBone('J_Bip_L_LowerArm'),
            rightElbow: findBone('J_Bip_R_LowerArm')
        };
        baseRotations = new Map();
        for (const bone of Object.values(idleBones)) {
            if (bone) baseRotations.set(bone, bone.getLocalRotation().clone());
        }
        updateTint();
    };

    const loadAvatar = async (url: string, label: string, objectUrl?: string) => {
        if (!app) throw new Error('PlayCanvas is not ready.');
        modelStatus.textContent = `${label} を読み込み中…`;
        const asset = new Asset(label, 'container', { url });
        app.assets.add(asset);
        try {
            await new Promise<void>((resolve, reject) => {
                asset.once('load', () => resolve());
                asset.once('error', (error: unknown) => reject(new Error(errorMessage(error))));
                app?.assets.load(asset);
            });
            if (disposed) {
                asset.unload();
                app.assets.remove(asset);
                return;
            }
            const entity = asset.resource?.instantiateRenderEntity();
            if (!entity) throw new Error('モデルから描画可能なシーンを作成できませんでした。');
            app.root.addChild(entity);
            setAvatar(entity);

            const previousAsset = currentAsset;
            currentAsset = asset;
            if (previousAsset && previousAsset !== asset) {
                previousAsset.unload();
                app.assets.remove(previousAsset);
            }
            if (uploadedObjectUrl && uploadedObjectUrl !== objectUrl) URL.revokeObjectURL(uploadedObjectUrl);
            uploadedObjectUrl = objectUrl;
            modelStatus.textContent = `${label} · PlayCanvas glTFコンテナ`;
            runtimeStatus.textContent = 'IDLE MOTION';
            addFeed(`アバター「${label}」を読み込みました`, '◈');
        } catch (error) {
            asset.unload();
            app.assets.remove(asset);
            if (objectUrl) URL.revokeObjectURL(objectUrl);
            modelStatus.textContent = `モデル読み込みエラー: ${errorMessage(error)}`;
            notify('モデルを読み込めませんでした。GLBコンテナ形式か確認してください。');
            throw error;
        }
    };

    const loadMotion = async () => {
        const response = await fetch('./stand2-idle.json');
        if (!response.ok) throw new Error(`アニメーションの取得に失敗しました (${response.status})`);
        motion = (await response.json()) as IdleMotion;
        if (!motion.fps || !motion.durationFrames || !motion.tracks) {
            throw new Error('アニメーションデータの形式が正しくありません。');
        }
    };

    const createFaceLandmarker = async () => {
        const moduleUrl = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/vision_bundle.mjs`;
        const wasmUrl = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`;
        const vision = (await import(/* @vite-ignore */ moduleUrl)) as VisionModule;
        const fileset = await vision.FilesetResolver.forVisionTasks(wasmUrl);
        const options = {
            baseOptions: { modelAssetPath: FACE_MODEL_URL, delegate: 'GPU' },
            runningMode: 'VIDEO',
            numFaces: 1,
            outputFacialTransformationMatrixes: true
        };
        try {
            return await vision.FaceLandmarker.createFromOptions(fileset, options);
        } catch {
            return vision.FaceLandmarker.createFromOptions(fileset, {
                ...options,
                baseOptions: { modelAssetPath: FACE_MODEL_URL, delegate: 'CPU' }
            });
        }
    };

    const updateFace = () => {
        if (!faceLandmarker || cameraPreview.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
        const now = performance.now();
        if (now - lastFaceDetection < 32) return;
        lastFaceDetection = now;

        const result = faceLandmarker.detectForVideo(cameraPreview, now);
        const matrix = result.facialTransformationMatrixes?.[0]?.data;
        if (!matrix || matrix.length < 16) {
            trackingPose = undefined;
            poseStatus.textContent = 'NO FACE';
            anglesReadout.textContent = 'X 0° · Y 0°';
            depthReadout.textContent = 'Z —';
            return;
        }

        const pose = matrixPose(matrix);
        if (!trackingBaseline) trackingBaseline = pose;
        trackingPose = {
            x: pose.x - trackingBaseline.x,
            y: pose.y - trackingBaseline.y,
            z: pose.z - trackingBaseline.z,
            yaw: pose.yaw,
            pitch: pose.pitch,
            roll: pose.roll
        };
        poseStatus.textContent = '6DOF ON';
        anglesReadout.textContent = `X ${Math.round((pose.yaw * 180) / Math.PI)}° · Y ${Math.round((pose.pitch * 180) / Math.PI)}°`;
        depthReadout.textContent = `Z ${trackingPose.z.toFixed(2)}`;
    };

    const trackFaceFrame = () => {
        if (disposed || !faceLandmarker) return;
        try {
            updateFace();
        } catch (error) {
            const detail = errorMessage(error);
            stopFaceTracking(false);
            trackingStatus.textContent = `追跡エラー: ${detail}`;
            notify(`Face Trackingを停止しました: ${detail}`);
            return;
        }
        trackingFrame = window.requestAnimationFrame(trackFaceFrame);
    };

    const stopFaceTracking = (announce = true) => {
        window.cancelAnimationFrame(trackingFrame);
        trackingFrame = 0;
        faceLandmarker?.close();
        faceLandmarker = undefined;
        trackingStream?.getTracks().forEach((track) => track.stop());
        trackingStream = undefined;
        cameraPreview.pause();
        cameraPreview.srcObject = null;
        cameraPreview.classList.add('hidden');
        cameraPlaceholder.classList.remove('hidden');
        cameraPlaceholder.textContent = 'カメラ追跡はオフです';
        cameraCopy.textContent = 'カメラは許可後に端末内で処理されます。';
        trackingPose = undefined;
        trackingBaseline = undefined;
        poseStatus.textContent = 'OFF';
        trackingButton.textContent = '◎ カメラで追跡';
        trackingButton.classList.remove('active');
        trackingStatus.textContent = 'カメラは開始操作後にだけ有効になります。';
        if (announce) addFeed('Face Trackingを停止しました', '◉');
    };

    const startFaceTracking = async () => {
        if (!navigator.mediaDevices?.getUserMedia) {
            throw new Error('このブラウザではカメラAPIを利用できません。HTTPSまたはlocalhostで開いてください。');
        }
        trackingButton.disabled = true;
        trackingStatus.textContent = 'カメラの許可を待っています…';
        try {
            trackingStream = await navigator.mediaDevices.getUserMedia({
                audio: false,
                video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } }
            });
            cameraPreview.srcObject = trackingStream;
            await cameraPreview.play();
            cameraPlaceholder.classList.add('hidden');
            cameraPreview.classList.remove('hidden');
            trackingStatus.textContent = 'Face Landmarkerを準備しています…';
            faceLandmarker = await createFaceLandmarker();
            if (disposed) {
                stopFaceTracking(false);
                return;
            }
            trackingButton.textContent = '■ 追跡を停止';
            trackingButton.classList.add('active');
            trackingStatus.textContent = '頭の移動・向きを読み取り、カメラ位置に反映します。';
            cameraCopy.textContent = '処理はこの端末のブラウザ内で実行中です。';
            poseStatus.textContent = 'SEARCHING';
            addFeed('端末内Face Trackingを開始しました', '◉');
            trackFaceFrame();
        } catch (error) {
            stopFaceTracking();
            const detail = errorMessage(error);
            trackingStatus.textContent = `追跡できませんでした: ${detail}`;
            cameraPlaceholder.textContent = 'カメラを利用できません';
            notify(`Face Trackingを開始できません: ${detail}`);
        } finally {
            trackingButton.disabled = false;
        }
    };

    const handlePointer = (event: PointerEvent) => {
        const bounds = appCanvas.getBoundingClientRect();
        pointerLook = {
            x: clamp(((event.clientX - bounds.left) / bounds.width - 0.5) * -1, -0.5, 0.5),
            y: clamp(((event.clientY - bounds.top) / bounds.height - 0.5) * -1, -0.5, 0.5)
        };
        interactionPulse = 1;
        runtimeStatus.textContent = 'HELLO!';
        window.setTimeout(() => {
            if (!disposed) runtimeStatus.textContent = motionEnabled ? 'IDLE MOTION' : 'MOTION PAUSED';
        }, 1600);
    };

    const openWebPreview = (urlText: string) => {
        let url: URL;
        try {
            url = new URL(urlText);
        } catch {
            throw new Error('正しいURLを入力してください。');
        }
        if (url.protocol !== 'https:' && url.protocol !== 'http:') {
            throw new Error('HTTPSまたはHTTPのURLのみ使用できます。');
        }
        if (url.hostname === 'youtube.com' || url.hostname === 'www.youtube.com') {
            const videoId = url.searchParams.get('v');
            if (videoId) url = new URL(`https://www.youtube-nocookie.com/embed/${encodeURIComponent(videoId)}`);
        } else if (url.hostname === 'youtu.be') {
            const videoId = url.pathname.slice(1);
            if (videoId) url = new URL(`https://www.youtube-nocookie.com/embed/${encodeURIComponent(videoId)}`);
        }
        webFrame.src = url.href;
        webFrame.hidden = false;
        webStatus.textContent = 'プレビューを開きました。外部サイトが埋め込みを拒否する場合があります。';
        addFeed(`Webプレビューを開きました: ${url.hostname}`, '↗');
    };

    const updateScene = (dt: number) => {
        if (!motionEnabled) return;
        animationTime += dt;
        interactionPulse = Math.max(0, interactionPulse - dt * 1.4);
        if (!currentAvatar) return;

        const elapsed = motion ? (animationTime * motion.fps) % motion.durationFrames : animationTime * 30;
        const duration = motion?.durationFrames ?? 600;
        const rootMotion = samplePosition(motion?.tracks.rootPosition, elapsed, duration);
        currentAvatar.setLocalPosition(
            rootMotion[0] * 0.45,
            rootMotion[1] + Math.sin(animationTime * 2.2) * 0.012 + interactionPulse * 0.12,
            rootMotion[2]
        );

        const tracks: [string, string][] = [
            ['hips', 'hipsRotation'],
            ['spine', 'spineRotation'],
            ['head', 'headRotation'],
            ['leftShoulder', 'leftShoulderRotation'],
            ['rightShoulder', 'rightShoulderRotation'],
            ['leftArm', 'leftArmRotation'],
            ['rightArm', 'rightArmRotation'],
            ['leftElbow', 'leftElbowRotation'],
            ['rightElbow', 'rightElbowRotation']
        ];
        for (const [boneName, trackName] of tracks) {
            const bone = idleBones[boneName];
            const base = bone && baseRotations.get(bone);
            if (!bone || !base) continue;
            const idle = sampleRotation(motion?.tracks[trackName], elapsed, duration);
            if (boneName === 'head') {
                const facePitch = trackingPose?.pitch ?? pointerLook.y * 0.24;
                const faceYaw = trackingPose?.yaw ?? pointerLook.x * 0.42;
                const faceRoll = trackingPose?.roll ?? 0;
                const look = new Quat().setFromEulerAngles(
                    (facePitch * 180) / Math.PI,
                    (faceYaw * 180) / Math.PI,
                    (faceRoll * 180) / Math.PI
                );
                bone.setLocalRotation(new Quat().mul2(new Quat().mul2(base, idle), look));
            } else {
                bone.setLocalRotation(new Quat().mul2(base, idle));
            }
        }

        if (cameraEntity && cameraBaseRotation) {
            if (trackingPose) {
                const targetPosition = [
                    -trackingPose.x * 1.6,
                    1.25 - trackingPose.y * 1.6,
                    4.5 + trackingPose.z * 2.2
                ] as const;
                const current = cameraEntity.getPosition();
                cameraEntity.setPosition(
                    current.x + (targetPosition[0] - current.x) * Math.min(1, dt * 5),
                    current.y + (targetPosition[1] - current.y) * Math.min(1, dt * 5),
                    current.z + (targetPosition[2] - current.z) * Math.min(1, dt * 5)
                );
                const viewRotation = new Quat().setFromEulerAngles(
                    -(trackingPose.pitch * 180 * 0.14) / Math.PI,
                    (trackingPose.yaw * 180 * 0.2) / Math.PI,
                    (trackingPose.roll * 180 * 0.18) / Math.PI
                );
                cameraEntity.setLocalRotation(new Quat().mul2(cameraBaseRotation, viewRotation));
            } else {
                const position = cameraEntity.getPosition();
                cameraEntity.setPosition(
                    position.x + (0 - position.x) * Math.min(1, dt * 2),
                    position.y + (1.25 - position.y) * Math.min(1, dt * 2),
                    position.z + (4.5 - position.z) * Math.min(1, dt * 2)
                );
                cameraEntity.setLocalRotation(cameraBaseRotation);
            }
        }
    };

    const appPromise = (async () => {
        const device = await createGraphicsDevice(appCanvas);
        if (disposed) {
            device.destroy();
            return;
        }
        device.maxPixelRatio = Math.min(window.devicePixelRatio, 2);
        const options = new AppOptions();
        options.graphicsDevice = device;
        options.componentSystems = [RenderComponentSystem, CameraComponentSystem, ScriptComponentSystem];
        options.resourceHandlers = [TextureHandler, ContainerHandler];
        app = new AppBase(appCanvas);
        app.init(options);
        app.setCanvasFillMode(FILLMODE_FILL_WINDOW);
        app.setCanvasResolution(RESOLUTION_AUTO);

        const sky = new Entity('sky');
        sky.addComponent('script');
        sky.script?.create(ProceduralSky, { properties: { luminance: 0.18 } });
        app.root.addChild(sky);

        cameraEntity = new Entity('camera');
        cameraEntity.addComponent('camera', { fov: 36 });
        cameraEntity.setPosition(0, 1.25, 4.5);
        cameraEntity.lookAt(0, 1.05, 0);
        cameraBaseRotation = cameraEntity.getLocalRotation().clone();
        app.root.addChild(cameraEntity);

        app.on('update', updateScene);
        app.start();

        await Promise.allSettled([loadAvatar('./avatar.glb', '付属アバター'), loadMotion()]).then((results) => {
            const modelResult = results[0];
            const motionResult = results[1];
            if (motionResult.status === 'rejected') {
                notify(
                    `VMD変換モーションを読み込めませんでした: ${errorMessage(motionResult.reason)}。簡易アイドル動作で続行します。`
                );
            }
            if (modelResult.status === 'rejected') throw modelResult.reason;
        });
    })();

    addMessage('assistant', 'こんにちは！ここをタッチすると反応するよ。カメラ追跡や音声も、設定から試してみてね。');
    addFeed('ミオがスタジオにやってきました');
    updateVoiceOptions();
    window.speechSynthesis?.addEventListener?.('voiceschanged', updateVoiceOptions);

    chatForm.addEventListener('submit', (event) => {
        event.preventDefault();
        void sendMessage(chatInput.value);
    });
    root.querySelectorAll<HTMLButtonElement>('[data-prompt]').forEach((button) => {
        button.addEventListener('click', () => void sendMessage(button.dataset.prompt ?? ''));
    });
    chatMode.addEventListener('change', updateChatMode);
    updateChatMode();

    byId<HTMLButtonElement>(root, 'greet-button').addEventListener('click', () => {
        void sendMessage('こんにちは、ミオ！');
    });
    appCanvas.addEventListener('pointerdown', handlePointer);
    motionButton.addEventListener('click', () => {
        motionEnabled = !motionEnabled;
        motionButton.textContent = motionEnabled ? 'Ⅱ モーション' : '▷ モーション';
        motionButton.setAttribute('aria-pressed', String(motionEnabled));
        runtimeStatus.textContent = motionEnabled ? 'IDLE MOTION' : 'MOTION PAUSED';
    });

    trackingButton.addEventListener('click', () => {
        if (faceLandmarker) stopFaceTracking();
        else void startFaceTracking();
    });

    byId<HTMLButtonElement>(root, 'model-button').addEventListener('click', () => modelInputFile.click());
    modelInputFile.addEventListener('change', () => {
        const file = modelInputFile.files?.[0];
        modelInputFile.value = '';
        if (!file) return;
        const extension = file.name.toLowerCase().split('.').pop();
        if (extension !== 'glb' && extension !== 'vrm') {
            notify('GLBコンテナ形式の .glb または .vrm ファイルを選択してください。');
            return;
        }
        if (file.size > 50 * 1024 * 1024) {
            notify('モデルの上限サイズは50MBです。');
            return;
        }
        void file
            .arrayBuffer()
            .then((buffer) => {
                if (new TextDecoder().decode(buffer.slice(0, 4)) !== 'glTF') {
                    throw new Error('選択したVRM/GLBはバイナリGLBコンテナではありません。');
                }
                const objectUrl = URL.createObjectURL(new Blob([buffer], { type: 'model/gltf-binary' }));
                return loadAvatar(objectUrl, file.name, objectUrl);
            })
            .catch((error: unknown) => {
                modelStatus.textContent = `モデル読み込みエラー: ${errorMessage(error)}`;
                notify(errorMessage(error));
            });
    });

    root.querySelectorAll<HTMLButtonElement>('[data-tint]').forEach((button) => {
        button.addEventListener('click', () => {
            tintName = button.dataset.tint ?? 'original';
            root.querySelectorAll<HTMLButtonElement>('[data-tint]').forEach((option) => {
                option.setAttribute('aria-pressed', String(option === button));
            });
            updateTint();
            addFeed('マテリアルカラーを変更しました', '◈');
        });
    });

    byId<HTMLButtonElement>(root, 'speak-test').addEventListener('click', () =>
        speak('こんにちは。ミオだよ。今日もよろしくね。')
    );
    byId<HTMLButtonElement>(root, 'stop-speech').addEventListener('click', () => window.speechSynthesis?.cancel());
    voiceSelect.addEventListener('change', () => {
        selectedVoice = voiceSelect.value;
    });

    const speechWindow = window as Window & {
        SpeechRecognition?: SpeechRecognitionConstructor;
        webkitSpeechRecognition?: SpeechRecognitionConstructor;
    };
    const Recognition = speechWindow.SpeechRecognition ?? speechWindow.webkitSpeechRecognition;
    const micButton = byId<HTMLButtonElement>(root, 'mic-button');
    if (!Recognition) {
        micButton.disabled = true;
        micButton.title = 'このブラウザは音声認識に対応していません';
    } else {
        speechRecognition = new Recognition();
        speechRecognition.lang = 'ja-JP';
        speechRecognition.continuous = false;
        speechRecognition.interimResults = false;
        speechRecognition.onresult = (event) => {
            const result = event.results[event.results.length - 1];
            if (result?.isFinal) void sendMessage(result[0].transcript);
        };
        speechRecognition.onerror = (event) => notify(`音声認識エラー: ${event.error ?? 'マイクを確認してください。'}`);
        speechRecognition.onend = () => micButton.classList.remove('active');
        micButton.addEventListener('click', () => {
            if (!speechRecognition) return;
            try {
                speechRecognition.start();
                micButton.classList.add('active');
            } catch (error) {
                notify(`音声入力を開始できません: ${errorMessage(error)}`);
            }
        });
    }

    const enableNotifications = async () => {
        if (!('Notification' in window)) {
            notify('このブラウザは通知に対応していません。');
            return;
        }
        const permission = await Notification.requestPermission();
        notify(permission === 'granted' ? 'この画面のフィード通知を有効にしました。' : '通知は許可されませんでした。');
    };
    byId<HTMLButtonElement>(root, 'notification-button').addEventListener('click', () => void enableNotifications());
    byId<HTMLButtonElement>(root, 'notify-enable').addEventListener('click', () => void enableNotifications());

    const installButton = byId<HTMLButtonElement>(root, 'install-button');
    const onInstallPrompt = (event: Event) => {
        event.preventDefault();
        deferredInstallPrompt = event as InstallPromptEvent;
        installButton.hidden = false;
    };
    window.addEventListener('beforeinstallprompt', onInstallPrompt);
    installButton.addEventListener('click', async () => {
        if (!deferredInstallPrompt) {
            notify('ブラウザのメニューから「アプリをインストール」を選んでください。');
            return;
        }
        await deferredInstallPrompt.prompt();
        deferredInstallPrompt = undefined;
        installButton.hidden = true;
    });

    byId<HTMLButtonElement>(root, 'panel-toggle').addEventListener('click', () => panel.classList.toggle('expanded'));
    byId<HTMLButtonElement>(root, 'panel-close').addEventListener('click', () => panel.classList.remove('expanded'));
    byId<HTMLFormElement>(root, 'web-form').addEventListener('submit', (event) => {
        event.preventDefault();
        try {
            openWebPreview(byId<HTMLInputElement>(root, 'web-url').value.trim());
        } catch (error) {
            webStatus.textContent = errorMessage(error);
        }
    });
    byId<HTMLButtonElement>(root, 'web-clear').addEventListener('click', () => {
        webFrame.removeAttribute('src');
        webFrame.hidden = true;
        webStatus.textContent = '外部サイトの埋め込み可否は提供元の設定によります。';
    });

    void appPromise.catch((error: unknown) => {
        if (!disposed) {
            modelStatus.textContent = `PlayCanvasを初期化できませんでした: ${errorMessage(error)}`;
            notify(`PlayCanvasの初期化に失敗しました: ${errorMessage(error)}`);
        }
    });

    return () => {
        disposed = true;
        window.clearTimeout(toastTimer);
        window.removeEventListener('beforeinstallprompt', onInstallPrompt);
        window.speechSynthesis?.removeEventListener?.('voiceschanged', updateVoiceOptions);
        speechRecognition?.stop();
        stopFaceTracking(false);
        if (uploadedObjectUrl) URL.revokeObjectURL(uploadedObjectUrl);
        currentAsset?.unload();
        app?.destroy();
        root.replaceChildren();
    };
};
