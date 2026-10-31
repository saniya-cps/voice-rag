import os
import html
import gradio as gr
from pathlib import Path
from src.config import Config
from src.pipeline import VoiceRAGPipeline

# Initialize the pipeline globally (starts as an empty NotebookLM knowledge base)
print("==================================================")
print("Loading Voice-Enabled RAG Pipeline...")
print("==================================================")
pipeline = VoiceRAGPipeline()
pipeline.initialize(load_starter_demo=False)
print("Pipeline loaded. Ready for customer-uploaded datasets and notes.")

def render_sources_list_html():
    """Generates an aesthetic HTML card list of all user-uploaded sources."""
    sources = pipeline.get_sources_summary()
    total_chunks = pipeline.get_total_passages_count()

    if not sources:
        return (
            '<div class="empty-source-card">'
            '<div style="font-weight:700; margin-bottom:0.35rem;">Notebook is Empty</div>'
            '<div>Drop your PDFs, CSVs, TSVs, notes, or research papers above to create your personal knowledge base.</div>'
            '</div>'
        )

    cards = []
    for s in sources:
        name = html.escape(s["name"])
        ftype = html.escape(s["file_type"].upper().replace(".", "")) or "DOC"
        chunks = s["num_chunks"]
        cards.append(f"""
        <div class="source-card">
            <div class="source-card-info">
                <span class="source-type-pill">{ftype}</span>
                <span class="source-card-name" title="{name}">{name}</span>
            </div>
            <span class="source-card-badge">{chunks} Chunks</span>
        </div>
        """)

    header = f'<div class="sources-list-header">Active Sources ({len(sources)}) &bull; {total_chunks} Total Passages</div>'
    return header + "".join(cards)

def get_source_dropdown_choices():
    """Returns dropdown choices for individual source removal."""
    sources = pipeline.get_sources_summary()
    return [f"{s['name']} ({s['source_id']})" for s in sources]

def get_notebook_status_banner():
    """Returns the main chat banner indicating notebook state."""
    sources = pipeline.get_sources_summary()
    total_chunks = pipeline.get_total_passages_count()
    if not sources:
        return (
            '<div class="notebook-status-banner empty-banner">'
            '<strong>Notebook Status: Empty</strong><br>'
            'No sources have been uploaded yet. Add documents or notes in the Source Drawer on the left to start asking grounded questions.'
            '</div>'
        )
    return (
        f'<div class="notebook-status-banner active-banner">'
        f'<strong>Notebook Status: Active ({len(sources)} Sources Loaded &bull; {total_chunks} Chunks Indexed)</strong><br>'
        f'Questions are strictly grounded in your uploaded documents. Citations will reference your exact files.'
        f'</div>'
    )

def handle_add_files(file_objs):
    """Processes batch file uploads into the user's notebook."""
    if not file_objs:
        return (
            render_sources_list_html(),
            gr.update(choices=get_source_dropdown_choices()),
            get_notebook_status_banner(),
            "*No files were selected.*"
        )

    # Gradio might return a single string/file or a list
    if not isinstance(file_objs, list):
        file_objs = [file_objs]

    added_names = []
    total_new_chunks = 0
    for f in file_objs:
        path = f.name if hasattr(f, "name") else str(f)
        try:
            src, count = pipeline.add_source_file(path)
            added_names.append(src.name)
            total_new_chunks += src.num_chunks
        except Exception as e:
            print(f"[Upload Error on {path}]: {e}")

    sources_html = render_sources_list_html()
    dropdown_update = gr.update(choices=get_source_dropdown_choices())
    banner_html = get_notebook_status_banner()
    status_msg = f"*Successfully added {len(added_names)} source(s) ({total_new_chunks} passages indexed in FAISS + BM25):* {', '.join(added_names)}"
    return sources_html, dropdown_update, banner_html, status_msg

def handle_add_note(note_title, note_content):
    """Adds a direct text note into the notebook."""
    if not note_content or not note_content.strip():
        return (
            render_sources_list_html(),
            gr.update(choices=get_source_dropdown_choices()),
            get_notebook_status_banner(),
            "*Note text cannot be empty.*",
            ""
        )
    title = (note_title.strip() if note_title and note_title.strip() else "Quick Note") + ".txt"
    source, total = pipeline.add_source_text(note_content.strip(), title=title)
    sources_html = render_sources_list_html()
    dropdown_update = gr.update(choices=get_source_dropdown_choices())
    banner_html = get_notebook_status_banner()
    status_msg = f"*Added note '{title}' with {source.num_chunks} passage chunks to your notebook.*"
    return sources_html, dropdown_update, banner_html, status_msg, ""

def handle_remove_source(choice):
    """Removes a specific source document."""
    if not choice:
        return (
            render_sources_list_html(),
            gr.update(choices=get_source_dropdown_choices()),
            get_notebook_status_banner(),
            "*No source selected for removal.*"
        )
    # Extract source_id from 'Filename (src_12345678)'
    try:
        source_id = choice.split("(")[-1].replace(")", "").strip()
        remaining = pipeline.remove_source(source_id)
        sources_html = render_sources_list_html()
        dropdown_update = gr.update(choices=get_source_dropdown_choices(), value=None)
        banner_html = get_notebook_status_banner()
        status_msg = f"*Removed source. Notebook now contains {remaining} active passages.*"
        return sources_html, dropdown_update, banner_html, status_msg
    except Exception as e:
        return (
            render_sources_list_html(),
            gr.update(choices=get_source_dropdown_choices()),
            get_notebook_status_banner(),
            f"*Error removing source: {str(e)}*"
        )

def handle_clear_all():
    """Clears all sources from the notebook."""
    pipeline.clear_all_sources()
    sources_html = render_sources_list_html()
    dropdown_update = gr.update(choices=[], value=None)
    banner_html = get_notebook_status_banner()
    status_msg = "*Cleared all sources. Notebook is now clean.*"
    return sources_html, dropdown_update, banner_html, status_msg

def handle_load_starter():
    """Loads starter knowledge base."""
    source, count = pipeline.load_starter_pack()
    sources_html = render_sources_list_html()
    dropdown_update = gr.update(choices=get_source_dropdown_choices())
    banner_html = get_notebook_status_banner()
    status_msg = f"*Loaded starter sample pack '{source.name}' ({source.num_chunks} chunks).* "
    return sources_html, dropdown_update, banner_html, status_msg

def process_interaction(audio_file, text_input, api_key_input, voice_enabled):
    """Processes user voice or text input and generates grounded answer + telemetry."""
    if api_key_input and api_key_input.strip():
        pipeline.llm.set_api_key(api_key_input.strip())

    audio_path = audio_file if audio_file else None
    query_text = text_input.strip() if text_input else None

    if not audio_path and not query_text:
        return (
            "Please record speech using the microphone, upload an audio file, or enter a text query.",
            None,
            "No query submitted.",
            ""
        )

    try:
        answer, audio_out, profile, passages = pipeline.run(
            audio_path=audio_path,
            text_query=query_text,
            synthesize_voice=voice_enabled
        )

        telemetry_md = profile.summary_markdown()

        passages_md_list = []
        if passages:
            for idx, p in enumerate(passages, 1):
                pid = p.get("passage_id", "N/A")
                meta = p.get("metadata", {})
                source_name = meta.get("source_name") or meta.get("source") or pid
                rrf = p.get("rrf_score", 0.0)
                dense = p.get("dense_score", 0.0)
                bm25 = p.get("bm25_score", 0.0)
                text = p.get("text", "")
                passages_md_list.append(
                    f"**Source {idx}: [{source_name}]** &bull; Chunk ID: `{pid}` (RRF: `{rrf:.4f}` | Dense Cosine: `{dense:.3f}` | BM25: `{bm25:.2f}`)\n> {text}\n"
                )
            passages_inspector = "\n\n".join(passages_md_list)
        else:
            if pipeline.get_total_passages_count() == 0:
                passages_inspector = "*Your notebook is empty. Please upload sources to enable grounded retrieval.*"
            else:
                passages_inspector = "*No passages in your notebook met the relevance threshold. Anti-hallucination guardrail triggered strict refusal.*"

        return answer, audio_out, telemetry_md, passages_inspector

    except Exception as exc:
        import traceback
        err_trace = traceback.format_exc()
        print(f"[ERROR in process_interaction]: {err_trace}")
        error_msg = "**System Alert:** An error occurred while processing your request. Please ensure your audio is clear or verify your Gemini API key."
        return error_msg, None, "*Request could not be completed.*", "*Check local server logs for details.*"


# Pure CSS for NotebookLM Studio + Pookie / Girl Coder Aesthetic
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Dynamic Ambient Glow Orbs */
@keyframes floatOrb1 {
    0% { transform: translate(0px, 0px) scale(1); }
    50% { transform: translate(50px, 35px) scale(1.12); }
    100% { transform: translate(-25px, 60px) scale(0.96); }
}

@keyframes floatOrb2 {
    0% { transform: translate(0px, 0px) scale(1); }
    50% { transform: translate(-40px, -35px) scale(1.15); }
    100% { transform: translate(35px, -50px) scale(0.92); }
}

@keyframes pulseGlow {
    0%, 100% { opacity: 0.45; }
    50% { opacity: 0.75; }
}

.pookie-bg-container {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    overflow: hidden;
    pointer-events: none;
    z-index: 0;
}

.pookie-orb {
    position: absolute;
    border-radius: 50%;
    filter: blur(100px);
    opacity: 0.6;
    animation-timing-function: ease-in-out;
    animation-iteration-count: infinite;
    animation-direction: alternate;
}

/* Dark Mode Orbs (Pink & Purple Aura) */
body.dark .orb-1 {
    width: 480px;
    height: 480px;
    top: -80px;
    left: 8%;
    background: radial-gradient(circle, rgba(255, 46, 147, 0.35) 0%, rgba(244, 63, 142, 0.15) 50%, transparent 80%);
    animation: floatOrb1 16s infinite alternate;
}

body.dark .orb-2 {
    width: 520px;
    height: 520px;
    bottom: -60px;
    right: 5%;
    background: radial-gradient(circle, rgba(168, 85, 247, 0.28) 0%, rgba(217, 70, 239, 0.12) 60%, transparent 80%);
    animation: floatOrb2 20s infinite alternate;
}

body.dark .orb-3 {
    width: 360px;
    height: 360px;
    top: 40%;
    left: 35%;
    background: radial-gradient(circle, rgba(255, 105, 180, 0.18) 0%, rgba(139, 92, 246, 0.08) 60%, transparent 80%);
    animation: pulseGlow 10s infinite;
}

/* Light Mode Orbs (Matcha Green & Sage Aura) */
body:not(.dark) .orb-1 {
    width: 480px;
    height: 480px;
    top: -80px;
    left: 8%;
    background: radial-gradient(circle, rgba(16, 185, 129, 0.25) 0%, rgba(52, 211, 153, 0.1) 60%, transparent 80%);
    animation: floatOrb1 16s infinite alternate;
}

body:not(.dark) .orb-2 {
    width: 520px;
    height: 520px;
    bottom: -60px;
    right: 5%;
    background: radial-gradient(circle, rgba(5, 150, 105, 0.2) 0%, rgba(110, 231, 183, 0.08) 60%, transparent 80%);
    animation: floatOrb2 20s infinite alternate;
}

body:not(.dark) .orb-3 {
    width: 360px;
    height: 360px;
    top: 40%;
    left: 35%;
    background: radial-gradient(circle, rgba(52, 211, 153, 0.15) 0%, rgba(16, 185, 129, 0.06) 60%, transparent 80%);
    animation: pulseGlow 10s infinite;
}

/* Base Structure */
* {
    box-sizing: border-box;
    transition: background-color 0.2s ease, border-color 0.2s ease, color 0.2s ease;
}

body, .gradio-container {
    font-family: 'Outfit', 'Inter', -apple-system, sans-serif !important;
    max-width: 1440px !important;
    margin: 0 auto !important;
    padding: 1.5rem !important;
    position: relative;
    z-index: 1;
}

/* Dark Mode Palette */
body.dark, html.dark {
    background-color: #08080d !important;
    color: #ffffff !important;
}

body.dark .gradio-container {
    background-color: #08080d !important;
    color: #ffffff !important;
}

body.dark .gr-panel,
body.dark .gr-box,
body.dark .gr-form,
body.dark .block,
body.dark .accordion {
    background: rgba(18, 17, 28, 0.8) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 77, 166, 0.28) !important;
    border-radius: 18px !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.55), inset 0 1px 0 rgba(255, 255, 255, 0.06) !important;
    color: #ffffff !important;
    padding: 1rem !important;
}

body.dark input,
body.dark textarea,
body.dark .gr-input {
    background: rgba(26, 24, 40, 0.9) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 77, 166, 0.3) !important;
    border-radius: 12px !important;
}

body.dark input:focus,
body.dark textarea:focus {
    border-color: #ff2e93 !important;
    box-shadow: 0 0 0 3px rgba(255, 46, 147, 0.45) !important;
}

/* Audio in Dark Mode */
body.dark .audio-container,
body.dark .audio-io,
body.dark .audio-player,
body.dark .waveform-container {
    background: rgba(26, 24, 40, 0.9) !important;
    border: 1px solid rgba(255, 77, 166, 0.3) !important;
    border-radius: 14px !important;
}

body.dark .record-button,
body.dark button.record,
body.dark .source-selection button,
body.dark select,
body.dark .dropdown {
    background: rgba(36, 32, 54, 0.95) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 77, 166, 0.35) !important;
    border-radius: 10px !important;
}

body.dark button.clear-button,
body.dark button.close,
body.dark button[aria-label="Clear"] {
    background: transparent !important;
    border: none !important;
    color: #a1a1aa !important;
}

body.dark .field-label {
    color: #ff80bf !important;
}

body.dark #submit-btn {
    background: linear-gradient(135deg, #ff2e93 0%, #ff6584 50%, #ec4899 100%) !important;
    box-shadow: 0 4px 20px rgba(255, 46, 147, 0.45) !important;
}

body.dark #submit-btn:hover {
    box-shadow: 0 8px 30px rgba(255, 46, 147, 0.7) !important;
}

/* Light Mode Palette */
body:not(.dark), html:not(.dark) {
    background-color: #fafaf9 !important;
    color: #09090b !important;
}

body:not(.dark) .gradio-container {
    background-color: #fafaf9 !important;
    color: #09090b !important;
}

body:not(.dark) .gr-panel,
body:not(.dark) .gr-box,
body:not(.dark) .gr-form,
body:not(.dark) .block,
body:not(.dark) .accordion {
    background: rgba(255, 255, 255, 0.92) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(16, 185, 129, 0.25) !important;
    border-radius: 18px !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.04), inset 0 1px 0 rgba(255, 255, 255, 0.8) !important;
    color: #09090b !important;
    padding: 1rem !important;
}

body:not(.dark) input,
body:not(.dark) textarea,
body:not(.dark) .gr-input {
    background: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 12px !important;
}

body:not(.dark) input:focus,
body:not(.dark) textarea:focus {
    border-color: #059669 !important;
    box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.3) !important;
}

body:not(.dark) .audio-container,
body:not(.dark) .audio-io,
body:not(.dark) .audio-player,
body:not(.dark) .waveform-container {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
}

body:not(.dark) .record-button,
body:not(.dark) button.record,
body:not(.dark) .source-selection button,
body:not(.dark) select,
body:not(.dark) .dropdown {
    background: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06) !important;
}

body:not(.dark) button.clear-button,
body:not(.dark) button.close,
body:not(.dark) button[aria-label="Clear"] {
    background: transparent !important;
    border: none !important;
    color: #64748b !important;
}

body:not(.dark) .field-label {
    color: #047857 !important;
}

body:not(.dark) #submit-btn {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
    box-shadow: 0 4px 20px rgba(16, 185, 129, 0.4) !important;
}

body:not(.dark) #submit-btn:hover {
    box-shadow: 0 8px 30px rgba(16, 185, 129, 0.6) !important;
}

body:not(.dark) .gr-markdown,
body:not(.dark) .gr-markdown p,
body:not(.dark) .gr-markdown td,
body:not(.dark) .gr-markdown th {
    color: #0f172a !important;
}
body:not(.dark) .gr-markdown em {
    color: #475569 !important;
}

/* Eradicate Gradio Floating Labels */
.gradio-container .block-label,
.gradio-container label.block-label,
.gradio-container span[data-testid="block-info"],
.gradio-container .label-wrap {
    display: none !important;
}

.field-label {
    font-size: 0.92rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    margin-bottom: 0.45rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* =========================================================================
   NOTEBOOKLM STUDIO SOURCE CARDS & BANNERS
   ========================================================================= */
.source-card {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem 0.95rem;
    margin-bottom: 0.55rem;
    border-radius: 12px;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

body.dark .source-card {
    background: rgba(28, 25, 42, 0.85);
    border: 1px solid rgba(255, 77, 166, 0.25);
}
body.dark .source-card:hover {
    border-color: #ff2e93;
    transform: translateX(3px);
}

body:not(.dark) .source-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
}
body:not(.dark) .source-card:hover {
    border-color: #10b981;
    transform: translateX(3px);
}

.source-card-info {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    overflow: hidden;
}

.source-type-pill {
    font-size: 0.7rem;
    font-weight: 800;
    padding: 0.2rem 0.5rem;
    border-radius: 6px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

body.dark .source-type-pill {
    background: rgba(255, 46, 147, 0.25);
    color: #ff80bf;
}
body:not(.dark) .source-type-pill {
    background: rgba(16, 185, 129, 0.2);
    color: #047857;
}

.source-card-name {
    font-size: 0.88rem;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 170px;
}

.source-card-badge {
    font-size: 0.76rem;
    font-weight: 700;
    padding: 0.25rem 0.65rem;
    border-radius: 9999px;
    white-space: nowrap;
}

body.dark .source-card-badge {
    background: rgba(255, 46, 147, 0.18);
    color: #ff80bf;
    border: 1px solid rgba(255, 46, 147, 0.35);
}
body:not(.dark) .source-card-badge {
    background: rgba(16, 185, 129, 0.12);
    color: #047857;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.empty-source-card {
    border-radius: 14px;
    padding: 1.5rem 1rem;
    text-align: center;
    font-size: 0.88rem;
    line-height: 1.5;
    margin-bottom: 0.75rem;
}

body.dark .empty-source-card {
    border: 2px dashed rgba(255, 77, 166, 0.3);
    color: #a1a1aa;
    background: rgba(22, 19, 34, 0.5);
}
body:not(.dark) .empty-source-card {
    border: 2px dashed #cbd5e1;
    color: #64748b;
    background: rgba(248, 250, 252, 0.7);
}

.sources-list-header {
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.65rem;
    opacity: 0.85;
}

.notebook-status-banner {
    border-radius: 14px;
    padding: 0.9rem 1.15rem;
    font-size: 0.9rem;
    line-height: 1.5;
    margin-bottom: 1.25rem;
}

body.dark .notebook-status-banner.empty-banner {
    background: rgba(255, 46, 147, 0.08);
    border: 1px solid rgba(255, 46, 147, 0.3);
    color: #fda4af;
}
body.dark .notebook-status-banner.active-banner {
    background: rgba(168, 85, 247, 0.12);
    border: 1px solid rgba(217, 70, 239, 0.35);
    color: #e9d5ff;
}

body:not(.dark) .notebook-status-banner.empty-banner {
    background: #fff1f2;
    border: 1px solid #fecdd3;
    color: #9f1239;
}
body:not(.dark) .notebook-status-banner.active-banner {
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    color: #065f46;
}

#submit-btn {
    color: #ffffff !important;
    border: none !important;
    border-radius: 14px !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    padding: 0.95rem 1.75rem !important;
    cursor: pointer !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    width: 100% !important;
}
#submit-btn:hover {
    transform: translateY(-2px);
    filter: brightness(1.08);
}

#theme-toggle-btn {
    border-radius: 9999px !important;
    padding: 0.6rem 1.35rem !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    backdrop-filter: blur(10px) !important;
}

body.dark #theme-toggle-btn {
    background: rgba(26, 24, 40, 0.85) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 77, 166, 0.3) !important;
}
body:not(.dark) #theme-toggle-btn {
    background: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06) !important;
}

.app-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 1.5rem;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid rgba(150, 150, 150, 0.15);
}

.title-group h1 {
    margin: 0 !important;
    font-size: 2.1rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
}

body.dark .title-group h1 {
    background: linear-gradient(135deg, #ffffff 40%, #ff80bf 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
body:not(.dark) .title-group h1 {
    background: linear-gradient(135deg, #09090b 40%, #059669 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}

.title-group p {
    margin: 0.35rem 0 0 0 !important;
    font-size: 0.95rem !important;
    opacity: 0.82;
}

.status-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.85rem;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.35rem 0.85rem;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    backdrop-filter: blur(10px);
}

body.dark .status-badge {
    background: rgba(255, 46, 147, 0.16);
    color: #ff80bf;
    border: 1px solid rgba(255, 46, 147, 0.4);
}
body:not(.dark) .status-badge {
    background: rgba(16, 185, 129, 0.12);
    color: #065f46;
    border: 1px solid rgba(16, 185, 129, 0.35);
}

/* Floating Glassmorphic Toast Notifications */
#pookie-toast {
    position: fixed;
    top: 24px;
    right: 24px;
    z-index: 999999;
    padding: 12px 20px;
    border-radius: 14px;
    font-size: 0.92rem;
    font-weight: 600;
    display: none;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    backdrop-filter: blur(18px);
    animation: slideInToast 0.35s cubic-bezier(0.16, 1, 0.3, 1);
}

body.dark #pookie-toast {
    background: rgba(26, 24, 40, 0.95);
    color: #ffffff;
    border: 1px solid #ff2e93;
    box-shadow: 0 0 20px rgba(255, 46, 147, 0.4);
}

body:not(.dark) #pookie-toast {
    background: rgba(255, 255, 255, 0.95);
    color: #0f172a;
    border: 1px solid #10b981;
    box-shadow: 0 8px 24px rgba(16, 185, 129, 0.25);
}

@keyframes slideInToast {
    from { transform: translateY(-30px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
}

table {
    width: 100% !important;
    border-collapse: collapse !important;
    margin: 1rem 0 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

th, td {
    padding: 0.75rem 1rem !important;
}

body.dark table, body.dark th, body.dark td {
    border: 1px solid rgba(255, 77, 166, 0.28) !important;
}
body.dark th { background-color: rgba(26, 24, 40, 0.85) !important; }

body:not(.dark) table, body:not(.dark) th, body:not(.dark) td {
    border: 1px solid #e2e8f0 !important;
}
body:not(.dark) th { background-color: #f1f5f9 !important; }

@media (max-width: 900px) {
    body, .gradio-container { padding: 0.75rem !important; }
    .app-header { flex-direction: column !important; align-items: flex-start !important; gap: 1rem !important; }
    #theme-toggle-btn { width: 100% !important; }
}
"""

# Client-side Theme Switcher, Confetti, Toast & Haptics
CLIENT_JS = """
function initPookieApp() {
    window.showPookieToast = function(msg) {
        let toast = document.getElementById('pookie-toast');
        if (!toast) {
            toast = document.createElement('div');
            toast.id = 'pookie-toast';
            document.body.appendChild(toast);
        }
        toast.innerText = msg;
        toast.style.display = 'block';
        if (navigator.vibrate) { navigator.vibrate([25, 35, 25]); }
        setTimeout(() => { toast.style.display = 'none'; }, 3200);
    };

    window.launchConfetti = function() {
        if (typeof confetti === 'function') {
            confetti({
                particleCount: 80,
                spread: 70,
                origin: { y: 0.7 },
                colors: ['#ff2e93', '#ff6584', '#ec4899', '#10b981', '#38bdf8', '#fbbf24']
            });
        }
    };

    function applyTheme(mode) {
        const isDark = (mode === 'dark');
        if (isDark) {
            document.documentElement.classList.add('dark');
            document.body.classList.add('dark');
            document.documentElement.classList.remove('light');
            document.body.classList.remove('light');
        } else {
            document.documentElement.classList.remove('dark');
            document.body.classList.remove('dark');
            document.documentElement.classList.add('light');
            document.body.classList.add('light');
        }

        const gradioApp = document.querySelector('gradio-app');
        if (gradioApp) {
            if (isDark) gradioApp.classList.add('dark');
            else gradioApp.classList.remove('dark');
        }

        localStorage.setItem('voice_rag_theme', mode);
        localStorage.setItem('theme', mode);

        const btn = document.getElementById('theme-toggle-btn');
        if (btn) {
            btn.innerText = isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode';
        }
    }

    const saved = localStorage.getItem('voice_rag_theme') || 'dark';
    applyTheme(saved);

    window.toggleThemeMode = function() {
        const isDark = document.body.classList.contains('dark') || document.documentElement.classList.contains('dark');
        const next = isDark ? 'light' : 'dark';
        applyTheme(next);
        window.showPookieToast(next === 'dark' ? 'Pookie Pink Dark Mode Activated' : 'Matcha Green Light Mode Activated');
    };
}
"""

HEAD_TAGS = """
<script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.9.4/dist/confetti.browser.min.js"></script>
"""

with gr.Blocks(title="Voice-Enabled RAG System") as demo:
    # Ambient Dynamic Glow Orbs & Toast
    gr.HTML("""
    <div class="pookie-bg-container">
        <div class="pookie-orb orb-1"></div>
        <div class="pookie-orb orb-2"></div>
        <div class="pookie-orb orb-3"></div>
    </div>
    <div id="pookie-toast"></div>
    """)

    # Top Header Bar with Theme Switcher
    with gr.Row(elem_classes=["app-header"]):
        with gr.Column(scale=4):
            gr.HTML("""
            <div class="title-group">
                <h1>Voice-Enabled RAG System</h1>
                <p>Curate Your Personal Knowledge Base | Strictly Grounded Audio & Text RAG | Sub-200ms Engine</p>
                <div class="status-badges">
                    <span class="status-badge">Personal Source Library</span>
                    <span class="status-badge">Source-Level Citations</span>
                    <span class="status-badge">Zero-Hallucination Guardrail</span>
                    <span class="status-badge">English & Hindi Multilingual</span>
                </div>
            </div>
            """)
        with gr.Column(scale=1, min_width=190):
            theme_btn = gr.Button("Switch Theme", elem_id="theme-toggle-btn")

    # Main Studio 2-Column Layout
    with gr.Row():
        # LEFT COLUMN: Source Knowledge Base (NotebookLM Drawer)
        with gr.Column(scale=4, min_width=350):
            gr.HTML('<div class="field-label" style="font-size:1.05rem;">Notebook Sources</div>')

            # Multi-File Dropzone
            file_uploader = gr.File(
                file_types=[".tsv", ".csv", ".pdf", ".txt", ".md"],
                file_count="multiple",
                show_label=False
            )
            add_files_btn = gr.Button("Add Documents to Notebook", variant="primary")

            # Quick Note Drawer
            with gr.Accordion("Paste Custom Research Note / Snippet", open=False):
                note_title_box = gr.Textbox(placeholder="Note Title (e.g., Biology Notes)", show_label=False)
                note_content_box = gr.Textbox(placeholder="Paste text notes, facts, or bullet points here...", lines=4, show_label=False)
                add_note_btn = gr.Button("Add Note to Notebook", variant="secondary")

            # Active Sources Cards Display
            gr.HTML('<div class="field-label" style="margin-top:1.2rem;">Source Library</div>')
            sources_display = gr.HTML(value=render_sources_list_html())

            # Source Management Controls
            with gr.Row():
                remove_dropdown = gr.Dropdown(
                    choices=get_source_dropdown_choices(),
                    label="Select Source to Remove",
                    show_label=False,
                    scale=2
                )
                remove_source_btn = gr.Button("Remove", variant="secondary", scale=1)

            with gr.Row():
                starter_btn = gr.Button("Load Demo Starter Pack", variant="secondary", scale=1)
                clear_btn = gr.Button("Clear All Sources", variant="secondary", scale=1)

            source_status_box = gr.Markdown(value="*Upload your documents above to populate your personal notebook.*")

            # Settings Accordion
            with gr.Accordion("Studio Settings & API Key", open=False):
                gr.HTML('<div class="field-label" style="font-size:0.85rem;">Gemini API Key</div>')
                api_key_box = gr.Textbox(
                    placeholder="Enter Gemini API key (or leave blank to use server .env)",
                    type="password",
                    value="",
                    show_label=False
                )
                voice_toggle = gr.Checkbox(label="Synthesize Voice Audio Response (TTS)", value=True)

        # RIGHT COLUMN: Grounded Chat & Voice Canvas
        with gr.Column(scale=6, min_width=450):
            # Dynamic Notebook State Banner
            notebook_banner = gr.HTML(value=get_notebook_status_banner())

            # Voice & Text Interaction Area
            gr.HTML('<div class="field-label">Voice Question (Microphone / Audio Upload)</div>')
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                show_label=False
            )

            gr.HTML('<div class="field-label" style="margin-top:0.75rem;">Text Question (English or Hindi)</div>')
            text_input = gr.Textbox(
                placeholder="Ask a question grounded in your uploaded documents...",
                lines=2,
                show_label=False
            )

            submit_btn = gr.Button("Ask Grounded Question", elem_id="submit-btn", variant="primary")

            # Grounded Output Answer & Neural Audio Player
            gr.HTML('<div class="field-label" style="margin-top:1.25rem;">Grounded Response (With Source Citations)</div>')
            output_answer = gr.Markdown(value="*Your grounded answer and document citations will appear here.*")
            output_audio = gr.Audio(show_label=False, autoplay=True)

            # Telemetry Metrics
            with gr.Accordion("Sub-200ms Latency Telemetry Breakdown", open=False):
                output_telemetry = gr.Markdown(value="*Latency telemetry breakdown will appear here upon submission.*")

            # Retrieved Passages Inspector
            with gr.Accordion("Retrieved Source Passages (Ground Truth Evidence)", open=False):
                output_passages = gr.Markdown(value="*Retrieved chunks with similarity metrics will be listed here.*")

    # Wire Interactions
    theme_btn.click(
        fn=None,
        js="() => { if (window.toggleThemeMode) { window.toggleThemeMode(); } }"
    )

    # Add Files to Notebook
    add_files_btn.click(
        fn=None,
        js="() => { if (window.launchConfetti) { window.launchConfetti(); } if (window.showPookieToast) { window.showPookieToast('Indexing uploaded documents into notebook...'); } }"
    ).then(
        fn=handle_add_files,
        inputs=[file_uploader],
        outputs=[sources_display, remove_dropdown, notebook_banner, source_status_box]
    )

    # Add Note to Notebook
    add_note_btn.click(
        fn=None,
        js="() => { if (window.launchConfetti) { window.launchConfetti(); } if (window.showPookieToast) { window.showPookieToast('Adding note to personal knowledge base...'); } }"
    ).then(
        fn=handle_add_note,
        inputs=[note_title_box, note_content_box],
        outputs=[sources_display, remove_dropdown, notebook_banner, source_status_box, note_content_box]
    )

    # Remove Source
    remove_source_btn.click(
        fn=None,
        js="() => { if (window.showPookieToast) { window.showPookieToast('Removing selected source from notebook...'); } }"
    ).then(
        fn=handle_remove_source,
        inputs=[remove_dropdown],
        outputs=[sources_display, remove_dropdown, notebook_banner, source_status_box]
    )

    # Clear All Sources
    clear_btn.click(
        fn=None,
        js="() => { if (window.showPookieToast) { window.showPookieToast('Cleared all sources from notebook.'); } }"
    ).then(
        fn=handle_clear_all,
        inputs=[],
        outputs=[sources_display, remove_dropdown, notebook_banner, source_status_box]
    )

    # Load Starter Demo Pack
    starter_btn.click(
        fn=None,
        js="() => { if (window.launchConfetti) { window.launchConfetti(); } if (window.showPookieToast) { window.showPookieToast('Loaded starter knowledge base!'); } }"
    ).then(
        fn=handle_load_starter,
        inputs=[],
        outputs=[sources_display, remove_dropdown, notebook_banner, source_status_box]
    )

    # Submit Question
    submit_btn.click(
        fn=None,
        js="() => { if (window.launchConfetti) { window.launchConfetti(); } if (window.showPookieToast) { window.showPookieToast('Querying notebook knowledge base...'); } }"
    ).then(
        fn=process_interaction,
        inputs=[audio_input, text_input, api_key_box, voice_toggle],
        outputs=[output_answer, output_audio, output_telemetry, output_passages]
    )

def attach_github_pages_bridge(fastapi_app):
    """Attaches CORS-enabled local machine REST endpoints so the static GitHub Pages site
    (https://<username>.github.io/<repo>/) can execute queries on this local Python machine.
    """
    import base64
    from fastapi import Request
    from fastapi.responses import JSONResponse

    CORS_HEADERS = {
        "Access-Control-Allow-Origin": "https://saniya-cps.github.io",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Private-Network": "true",
    }

    @fastapi_app.options("/api/{rest_of_path:path}")
    async def preflight_handler(rest_of_path: str):
        return JSONResponse(content={"ok": True}, headers=CORS_HEADERS)

    @fastapi_app.get("/api/status")
    async def api_status():
        sources = []
        for s in pipeline.source_manager.sources.values():
            sources.append({
                "source_id": s.source_id,
                "name": s.name,
                "file_type": s.file_type,
                "num_chunks": s.num_chunks,
                "passages": [p.to_dict() for p in s.passages],
            })
        return JSONResponse(content={"online": True, "sources": sources}, headers=CORS_HEADERS)

    @fastapi_app.post("/api/starter")
    async def api_starter():
        src, total = pipeline.load_starter_pack()
        return JSONResponse(content={"ok": True, "total": total}, headers=CORS_HEADERS)

    @fastapi_app.post("/api/clear")
    async def api_clear():
        pipeline.clear_all_sources()
        return JSONResponse(content={"ok": True}, headers=CORS_HEADERS)

    @fastapi_app.post("/api/add_note")
    async def api_add_note(req: Request):
        body = await req.json()
        title = body.get("title", "Uploaded_Source.txt")
        text = body.get("text", "")
        if text.strip():
            pipeline.add_source_text(text, title=title)
        return JSONResponse(content={"ok": True}, headers=CORS_HEADERS)

    @fastapi_app.post("/api/query")
    async def api_query(req: Request):
        body = await req.json()
        query = body.get("query", "")
        api_key = body.get("api_key", "")
        synthesize_voice = body.get("synthesize_voice", True)

        answer, audio_out, telemetry_md, passages_md = process_interaction(
            None, query, api_key, synthesize_voice
        )
        audio_b64 = None
        if audio_out and Path(audio_out).exists():
            audio_b64 = base64.b64encode(Path(audio_out).read_bytes()).decode("ascii")

        return JSONResponse(
            content={
                "answer": answer,
                "audio_base64": audio_b64,
                "telemetry_html": f"<pre style='white-space:pre-wrap;font-family:inherit;'>{html.escape(telemetry_md)}</pre>",
                "passages_html": f"<pre style='white-space:pre-wrap;font-family:inherit;'>{html.escape(passages_md)}</pre>",
            },
            headers=CORS_HEADERS,
        )


if __name__ == "__main__":
    print("==================================================")
    print("Starting Voice-Enabled RAG System Web Interface...")
    print("==================================================")
    is_hf_space = bool(os.getenv("SPACE_ID"))
    server_host = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0" if is_hf_space else "127.0.0.1")
    server_port = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", 7860)))
    launch_kwargs = {
        "server_name": server_host,
        "inbrowser": (not is_hf_space and server_host == "127.0.0.1"),
        "share": os.getenv("GRADIO_SHARE", "false").lower() == "true",
        "css": CUSTOM_CSS,
        "js": CLIENT_JS,
        "head": HEAD_TAGS,
        "prevent_thread_lock": True,
    }
    try:
        fastapi_app, local_url, share_url = demo.launch(server_port=server_port, **launch_kwargs)
    except OSError:
        print(f"[Notice] Port {server_port} is occupied. Launching on next available free port...")
        fastapi_app, local_url, share_url = demo.launch(**launch_kwargs)

    attach_github_pages_bridge(fastapi_app)
    demo.block_thread()
