import os
from graphviz import Digraph

def generate_short_deployment_diagram():
    # Initialize Digraph
    dot = Digraph('ThermalRGB_Short_Deployment', 
                 comment='Thermal-RGB 2D Detection Deployment',
                 format='png')
    
    # Global Attributes: Horizontal layout (LR)
    dot.attr(rankdir='LR', nodesep='0.4', ranksep='0.5')
    dot.attr(fontname='Segoe UI,Arial', fontsize='14', labelloc='t', 
             label='Deployment Pipeline: PyTorch → ONNX → TensorRT FP16')

    # Minimalist Styling
    dot.attr('node', fontname='Segoe UI Semibold,Arial', shape='box', style='filled', 
             color='#2d3436', fillcolor='#ecf0f1', penwidth='1.5')
    dot.attr('edge', fontname='Segoe UI,Arial', color='#636e72', penwidth='1.2', arrowsize='0.8')

    # --- Concise Nodes ---
    dot.node('ckpt', 'Checkpoint\n(.pth)', shape='cylinder', fillcolor='#fab1a0')
    dot.node('model', 'PyTorch Model\n(In-Memory)')
    dot.node('onnx', 'ONNX Export\n(.onnx)', fillcolor='#74b9ff')
    dot.node('builder', 'TRT Builder\n(FP16 + Dynamic)')
    dot.node('engine', 'TRT Engine\n(.engine)', shape='cylinder', fillcolor='#00b894', fontcolor='white')
    dot.node('deploy', 'Inference Wrapper\n(Python/C++)', fillcolor='#6c5ce7', fontcolor='white')

    # --- Linear Connections ---
    dot.edge('ckpt', 'model')
    dot.edge('model', 'onnx', label='torch.onnx.export')
    dot.edge('onnx', 'builder', label='Parse')
    dot.edge('builder', 'engine', label='Build')
    dot.edge('engine', 'deploy', label='Runtime')

    # Output directory
    results_dir = 'results/visualizations'
    os.makedirs(results_dir, exist_ok=True)
    
    output_path = os.path.join(results_dir, 'short_deployment_flow')
    try:
        dot.render(output_path, cleanup=True, view=False)
        print(f"✅ Short horizontal diagram saved to: {output_path}.png")
    except Exception as e:
        print(f"❌ Render failed: {e}")
        print("💡 Tip: Ensure the Graphviz system binary is installed.")

if __name__ == '__main__':
    generate_short_deployment_diagram()
