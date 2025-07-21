"""
PocketFlow-based Local PDF Chat RAG System
Converted from Local_Pdf_Chat.txt to use PocketFlow components
"""
import sys
import os
import gradio as gr
from flow import create_offline_flow, create_online_flow
from utils.api_config import check_environment, check_serpapi_key

def create_shared_store():
    """Create the shared data store for the entire RAG system"""
    return {
        # Document processing
        "files": [],
        "extracted_texts": [],
        "chunks": [],
        "chunk_metadatas": [],
        "chunk_ids": [],
        
        # Embedding and indexing
        "embeddings": None,
        "faiss_index": None,
        "faiss_contents_map": {},
        "faiss_metadatas_map": {},
        "faiss_id_order": [],
        
        # BM25 indexing  
        "bm25_manager": None,
        
        # Query processing
        "query": "",
        "query_embedding": None,
        "enable_web_search": False,
        "model_choice": "ollama",
        
        # Retrieval results
        "semantic_results": [],
        "bm25_results": [],
        "hybrid_results": [],
        "reranked_results": [],
        "web_results": [],
        
        # Final output
        "context": "",
        "answer": "",
        "sources": [],
        
        # Configuration
        "config": {
            "chunk_size": 400,
            "chunk_overlap": 40,
            "embedding_model": "all-MiniLM-L6-v2",
            "rerank_method": "cross_encoder",
            "hybrid_alpha": 0.7,
            "max_iterations": 3,
            "top_k": 10,
            "final_k": 5
        }
    }

def process_pdfs_and_answer(files, question, enable_web_search=False, model_choice="ollama"):
    """Main function to process PDFs and answer questions"""
    if not files:
        return "请先上传PDF文件", None, "❌ 没有上传文件"
    
    if not question.strip():
        return "请输入问题", None, "❌ 问题不能为空"
    
    # Create shared store
    shared = create_shared_store()
    shared["files"] = files
    shared["query"] = question
    shared["enable_web_search"] = enable_web_search
    shared["model_choice"] = model_choice
    
    try:
        # Create flows
        offline_flow = create_offline_flow()
        online_flow = create_online_flow()
        
        # Run offline flow (document processing and indexing)
        print("🔄 开始处理文档...")
        offline_flow.run(shared)
        
        # Run online flow (query processing and answer generation)
        print("🔍 开始回答问题...")
        online_flow.run(shared)
        
        # Format results
        answer = shared.get("answer", "未能生成回答")
        sources = shared.get("sources", [])
        
        # Create chat history format for Gradio (messages format)
        chat_history = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
        
        # Create processing status
        total_chunks = len(shared.get("chunks", []))
        processed_files = len([f for f in files if f is not None])
        status = f"✅ 成功处理 {processed_files} 个文件，{total_chunks} 个文本块"
        
        return chat_history, "", status
        
    except Exception as e:
        error_msg = f"处理过程出错: {str(e)}"
        print(f"❌ {error_msg}")
        error_chat = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"❌ {error_msg}"}
        ]
        return error_chat, "", f"❌ {error_msg}"

def create_gradio_interface():
    """Create the Gradio web interface"""
    
    with gr.Blocks(title="PocketFlow PDF Chat", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📚 PocketFlow PDF Chat RAG System")
        gr.Markdown("基于PocketFlow框架的本地PDF问答系统")
        
        with gr.Row():
            # Left panel - File upload and settings
            with gr.Column(scale=1):
                gr.Markdown("## 📁 文档上传")
                file_input = gr.File(
                    label="上传PDF文档",
                    file_types=[".pdf"],
                    file_count="multiple"
                )
                
                gr.Markdown("## ⚙️ 设置")
                web_search_checkbox = gr.Checkbox(
                    label="启用联网搜索", 
                    value=False,
                    info="需要配置SERPAPI_KEY"
                )
                
                model_choice = gr.Dropdown(
                    choices=["ollama", "siliconflow"],
                    value="ollama",
                    label="模型选择",
                    info="选择使用本地模型或云端模型"
                )
                
                gr.Markdown("## 🔍 提问")
                question_input = gr.Textbox(
                    label="输入问题",
                    lines=3,
                    placeholder="请输入您的问题..."
                )
                
                ask_btn = gr.Button("🚀 开始处理", variant="primary")
                
            # Right panel - Chat and status
            with gr.Column(scale=2):
                gr.Markdown("## 💬 对话")
                chatbot = gr.Chatbot(
                    label="对话历史",
                    height=400,
                    type="messages"
                )
                
                status_display = gr.Textbox(
                    label="处理状态",
                    interactive=False,
                    lines=2
                )
        
        # Environment status
        with gr.Row():
            gr.Markdown("### 🔧 环境状态")
            with gr.Column():
                env_status = gr.HTML()
                
                def check_env_status():
                    ollama_ok = check_environment()
                    serpapi_ok = check_serpapi_key()
                    
                    status_html = f"""
                    <div style="padding: 10px; border-radius: 5px; background: #f0f0f0;">
                        <p><strong>Ollama服务:</strong> {'✅ 正常' if ollama_ok else '❌ 异常'}</p>
                        <p><strong>SERPAPI配置:</strong> {'✅ 已配置' if serpapi_ok else '❌ 未配置'}</p>
                    </div>
                    """
                    return status_html
                
                demo.load(check_env_status, outputs=env_status)
        
        # Bind events
        ask_btn.click(
            fn=process_pdfs_and_answer,
            inputs=[file_input, question_input, web_search_checkbox, model_choice],
            outputs=[chatbot, question_input, status_display]
        )
    
    return demo

if __name__ == "__main__":
    # Check environment before starting
    if not check_environment():
        print("❌ 环境检查失败！请确保Ollama服务已启动")
        print("启动命令: ollama serve")
        print("模型安装: ollama pull deepseek-r1:1.5b")
        sys.exit(1)
    
    # Create and launch interface
    demo = create_gradio_interface()
    
    # Launch with auto port selection
    ports = [17995, 17996, 17997, 17998, 17999]
    for port in ports:
        try:
            demo.launch(
                server_port=port,
                server_name="0.0.0.0",
                show_error=True,
                share=False
            )
            break
        except Exception as e:
            if port == ports[-1]:  # Last port
                print(f"❌ 所有端口都被占用: {e}")
                sys.exit(1)
            continue
