#!/usr/bin/env python3
"""
知识图谱可视化

用法：
    python visualize_graph.py                    # 可视化 full_graph
    python visualize_graph.py --graph gis_graph  # 可视化 gis_graph
    python visualize_graph.py --output graph.html # 输出到指定文件
"""

import os
import json
import argparse
from typing import Dict, List, Any

# 设置离线模式
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"


def load_graph_data(graph_dir: str) -> Dict[str, Any]:
    """加载图谱数据"""
    import networkx as nx
    
    graphml_file = os.path.join(graph_dir, "graph_chunk_entity_relation.graphml")
    if not os.path.exists(graphml_file):
        print(f"图谱文件不存在: {graphml_file}")
        return None
    
    G = nx.read_graphml(graphml_file)
    
    nodes = []
    for node_id in G.nodes():
        node_data = G.nodes[node_id]
        nodes.append({
            "id": node_id,
            "type": node_data.get("entity_type", "Unknown"),
            "description": node_data.get("description", "")[:100]
        })

    edges = []
    for src, tgt, data in G.edges(data=True):
        edges.append({
            "source": src,
            "target": tgt,
            "type": data.get("keywords", "related").split(",")[0].strip(),
            "description": data.get("description", "")[:100]
        })

    # 预计算布局（固定随机种子，每次相同）
    pos = nx.spring_layout(G, seed=42, k=2.0, iterations=100)
    for node in nodes:
        node["x"] = float(pos[node["id"]][0])
        node["y"] = float(pos[node["id"]][1])

    return {"nodes": nodes, "edges": edges}


def generate_html_visualization(graph_data: Dict[str, Any], title: str = "知识图谱") -> str:
    """生成HTML可视化页面"""
    
    nodes_json = json.dumps(graph_data["nodes"], ensure_ascii=False)
    edges_json = json.dumps(graph_data["edges"], ensure_ascii=False)
    
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Microsoft YaHei', sans-serif; background: #1a1a2e; color: #eee; }}
        .container {{ display: flex; height: 100vh; }}
        .sidebar {{ width: 300px; background: #16213e; padding: 20px; overflow-y: auto; }}
        .graph-area {{ flex: 1; position: relative; }}
        #graph {{ width: 100%; height: 100%; }}
        .legend {{ position: absolute; top: 20px; right: 20px; background: rgba(22,33,62,0.9); padding: 15px; border-radius: 8px; }}
        .legend-item {{ display: flex; align-items: center; margin: 5px 0; }}
        .legend-color {{ width: 20px; height: 20px; border-radius: 50%; margin-right: 10px; }}
        .stats {{ background: #0f3460; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
        .node-info {{ background: #0f3460; padding: 15px; border-radius: 8px; margin-top: 20px; }}
        h2 {{ color: #e94560; margin-bottom: 15px; }}
        .search-box {{ width: 100%; padding: 10px; border: 1px solid #444; border-radius: 5px; background: #1a1a2e; color: #eee; margin-bottom: 15px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h2>🏙️ 城市变迁知识图谱</h2>
            
            <div class="stats">
                <h3>📊 图谱统计</h3>
                <p>节点数量: <strong>{len(graph_data['nodes'])}</strong></p>
                <p>关系数量: <strong>{len(graph_data['edges'])}</strong></p>
            </div>
            
            <input type="text" class="search-box" id="searchInput" placeholder="搜索节点...">
            
            <div class="legend">
                <h3>🎨 图例</h3>
                <div class="legend-item">
                    <div class="legend-color" style="background: #e94560;"></div>
                    <span>Boundary (边界)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #00b4d8;"></div>
                    <span>Point (点)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #06d6a0;"></div>
                    <span>Policy (政策)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #ffd166;"></div>
                    <span>STTE_Event (事件)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #bb86fc;"></div>
                    <span>District (区域)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #ff9a76;"></div>
                    <span>其他</span>
                </div>
            </div>
            
            <div class="node-info" id="nodeInfo">
                <h3>📝 节点信息</h3>
                <p id="nodeInfoText">点击节点查看详情</p>
            </div>
        </div>
        
        <div class="graph-area">
            <canvas id="graph"></canvas>
        </div>
    </div>
    
    <script>
        const nodes = {nodes_json};
        const edges = {edges_json};
        
        // 颜色映射
        const typeColors = {{
            'Boundary': '#e94560',
            'Point': '#00b4d8',
            'Policy': '#06d6a0',
            'STTE_Event': '#ffd166',
            'District': '#bb86fc',
            'PolicyGoal': '#ff6b6b',
            'PolicyMeasure': '#48dbfb',
            'Infrastructure': '#ff9a76',
            'LandUse': '#a8e6cf',
            'Keyword': '#dfe6e9',
            'DocumentChunk': '#b2bec3',
            'Unknown': '#636e72'
        }};
        
        // 初始化画布
        const canvas = document.getElementById('graph');
        const ctx = canvas.getContext('2d');
        let width, height;
        
        function resize() {{
            width = canvas.parentElement.clientWidth;
            height = canvas.parentElement.clientHeight;
            canvas.width = width;
            canvas.height = height;
        }}
        resize();
        window.addEventListener('resize', resize);
        
        // 节点布局（使用服务端预计算位置，叠加少量力导向微调后固定）
        const nodePositions = {{}};
        nodes.forEach(node => {{
            // 预计算坐标映射到画布
            const scaleX = width * 0.4;
            const scaleY = height * 0.4;
            nodePositions[node.id] = {{
                x: width/2 + (node.x || 0) * scaleX,
                y: height/2 + (node.y || 0) * scaleY,
                vx: 0,
                vy: 0
            }};
        }});

        // 力导向布局——有限迭代后停止
        const MAX_ITERATIONS = 300;
        let iteration = 0;
        let running = true;

        function applyForces() {{
            if (!running) return;
            const k = 0.005;
            const repulsion = 3000;

            // 节点间排斥
            for (let i = 0; i < nodes.length; i++) {{
                for (let j = i + 1; j < nodes.length; j++) {{
                    const n1 = nodePositions[nodes[i].id];
                    const n2 = nodePositions[nodes[j].id];
                    const dx = n2.x - n1.x;
                    const dy = n2.y - n1.y;
                    const dist = Math.sqrt(dx*dx + dy*dy) || 1;
                    const force = repulsion / (dist * dist);

                    n1.vx -= (dx / dist) * force;
                    n1.vy -= (dy / dist) * force;
                    n2.vx += (dx / dist) * force;
                    n2.vy += (dy / dist) * force;
                }}
            }}

            // 边的吸引力
            edges.forEach(edge => {{
                const n1 = nodePositions[edge.source];
                const n2 = nodePositions[edge.target];
                if (!n1 || !n2) return;

                const dx = n2.x - n1.x;
                const dy = n2.y - n1.y;
                const dist = Math.sqrt(dx*dx + dy*dy) || 1;
                const force = k * (dist - 100);

                n1.vx += (dx / dist) * force;
                n1.vy += (dy / dist) * force;
                n2.vx -= (dx / dist) * force;
                n2.vy -= (dy / dist) * force;
            }});

            // 更新位置
            nodes.forEach(node => {{
                const pos = nodePositions[node.id];
                pos.vx *= 0.85;  // 强阻尼——快速收敛
                pos.vy *= 0.85;
                pos.x += pos.vx;
                pos.y += pos.vy;

                pos.x = Math.max(30, Math.min(width - 30, pos.x));
                pos.y = Math.max(30, Math.min(height - 30, pos.y));
            }});

            iteration++;
            if (iteration >= MAX_ITERATIONS) {{
                running = false;
            }}
        }}

        // 动画循环——停止后固定
        function animate() {{
            if (running) {{
                applyForces();
            }}
            draw();
            requestAnimationFrame(animate);
        }}
        animate();
        
        // 绘制
        function draw() {{
            ctx.clearRect(0, 0, width, height);
            
            // 绘制边
            edges.forEach(edge => {{
                const n1 = nodePositions[edge.source];
                const n2 = nodePositions[edge.target];
                if (!n1 || !n2) return;
                
                ctx.beginPath();
                ctx.moveTo(n1.x, n1.y);
                ctx.lineTo(n2.x, n2.y);
                ctx.strokeStyle = 'rgba(255,255,255,0.15)';
                ctx.lineWidth = 1;
                ctx.stroke();
            }});
            
            // 绘制节点
            nodes.forEach(node => {{
                const pos = nodePositions[node.id];
                const color = typeColors[node.type] || typeColors['Unknown'];
                const radius = node.type === 'Boundary' ? 12 : 8;
                
                // 光晕效果
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, radius + 4, 0, Math.PI * 2);
                ctx.fillStyle = color + '40';
                ctx.fill();
                
                // 节点
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.fill();
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 2;
                ctx.stroke();
                
                // 标签
                ctx.fillStyle = '#fff';
                ctx.font = '11px Microsoft YaHei';
                ctx.textAlign = 'center';
                ctx.fillText(node.id.substring(0, 8), pos.x, pos.y + radius + 15);
            }});
        }}

        // 搜索功能
        document.getElementById('searchInput').addEventListener('input', (e) => {{
            const query = e.target.value.toLowerCase();
            nodes.forEach(node => {{
                const pos = nodePositions[node.id];
                if (query && node.id.toLowerCase().includes(query)) {{
                    pos.highlighted = true;
                }} else {{
                    pos.highlighted = false;
                }}
            }});
        }});
        
        // 点击节点
        canvas.addEventListener('click', (e) => {{
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            for (const node of nodes) {{
                const pos = nodePositions[node.id];
                const dist = Math.sqrt((x - pos.x)**2 + (y - pos.y)**2);
                if (dist < 15) {{
                    document.getElementById('nodeInfoText').innerHTML = `
                        <strong>${{node.id}}</strong><br>
                        类型: ${{node.type}}<br>
                        描述: ${{node.description || '无'}}
                    `;
                    break;
                }}
            }}
        }});
    </script>
</body>
</html>"""
    
    return html


def main():
    parser = argparse.ArgumentParser(description="知识图谱可视化")
    parser.add_argument("--graph", default="full_graph", choices=["gis_graph", "full_graph"], help="选择图谱")
    parser.add_argument("--open", action="store_true", help="自动打开浏览器")
    
    args = parser.parse_args()
    
    graph_dir = os.path.join("data", args.graph)
    
    print(f"加载图谱: {graph_dir}")
    graph_data = load_graph_data(graph_dir)
    
    if graph_data is None:
        print("图谱数据为空，请先导入数据")
        return
    
    print(f"节点: {len(graph_data['nodes'])}, 边: {len(graph_data['edges'])}")
    
    output = ''.join(['data/graph_visualization_', args.graph, '.html'])
    print(f"生成可视化: {output}")
    html = generate_html_visualization(graph_data, title=f"{args.graph} 知识图谱")
    
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"完成! 请用浏览器打开: {output}")
    
    if args.open:
        import webbrowser
        webbrowser.open(output)


if __name__ == "__main__":
    main()