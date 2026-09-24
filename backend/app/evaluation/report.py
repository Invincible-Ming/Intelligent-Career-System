"""
HTML 报告生成器。
"""

from __future__ import annotations

from typing import Any


class ReportGenerator:
    """HTML 报告生成器。"""

    def generate(
        self,
        report_data: dict[str, Any],
    ) -> str:
        """生成完整的 HTML 报告。"""

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG 评测报告 - {report_data['experiment_name']}</title>
    <style>
        {self._get_styles()}
    </style>
</head>
<body>
    <div class="container">
        {self._render_header(report_data)}
        {self._render_summary(report_data)}
        {self._render_metrics_comparison(report_data)}
        {self._render_detailed_results(report_data)}
    </div>
    <script>
        {self._get_scripts()}
    </script>
</body>
</html>"""

        return html

    def _get_styles(self) -> str:
        """获取 CSS 样式。"""

        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: #f5f7fa;
            color: #2c3e50;
            line-height: 1.6;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .header h1 {
            font-size: 32px;
            margin-bottom: 10px;
        }

        .header .meta {
            opacity: 0.9;
            font-size: 14px;
        }

        .section {
            background: white;
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }

        .section-title {
            font-size: 24px;
            margin-bottom: 20px;
            color: #667eea;
            border-bottom: 2px solid #f0f0f0;
            padding-bottom: 10px;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .stat-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }

        .stat-label {
            font-size: 14px;
            color: #666;
            margin-bottom: 8px;
        }

        .stat-value {
            font-size: 32px;
            font-weight: bold;
            color: #2c3e50;
        }

        .config-tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid #f0f0f0;
        }

        .tab {
            padding: 12px 24px;
            cursor: pointer;
            border: none;
            background: transparent;
            font-size: 16px;
            color: #666;
            border-bottom: 3px solid transparent;
            transition: all 0.3s;
        }

        .tab:hover {
            color: #667eea;
        }

        .tab.active {
            color: #667eea;
            border-bottom-color: #667eea;
            font-weight: bold;
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }

        .metrics-table th,
        .metrics-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #f0f0f0;
        }

        .metrics-table th {
            background: #f5f7fa;
            font-weight: 600;
            color: #2c3e50;
        }

        .metrics-table tr:hover {
            background: #f9fafb;
        }

        .metric-bar {
            height: 20px;
            background: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
            margin-top: 5px;
        }

        .metric-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.5s;
        }

        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }

        .badge-success {
            background: #d4edda;
            color: #155724;
        }

        .badge-danger {
            background: #f8d7da;
            color: #721c24;
        }

        .badge-warning {
            background: #fff3cd;
            color: #856404;
        }

        .test-result {
            background: #f9fafb;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 4px;
        }

        .test-result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }

        .test-query {
            font-weight: 600;
            color: #2c3e50;
            flex: 1;
        }

        .test-metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }

        .test-metric {
            background: white;
            padding: 8px;
            border-radius: 4px;
            font-size: 13px;
        }

        .test-metric-label {
            color: #666;
            font-size: 11px;
        }

        .test-metric-value {
            font-weight: bold;
            font-size: 16px;
            color: #667eea;
        }

        .answer-box {
            background: white;
            padding: 12px;
            border-radius: 4px;
            margin-top: 10px;
            font-size: 14px;
            line-height: 1.6;
        }

        .answer-label {
            font-weight: 600;
            color: #666;
            margin-bottom: 5px;
        }

        .comparison-chart {
            margin: 30px 0;
        }

        .chart-bar {
            margin-bottom: 15px;
        }

        .chart-label {
            font-size: 14px;
            color: #666;
            margin-bottom: 5px;
        }

        .chart-bars {
            display: flex;
            gap: 10px;
            align-items: center;
        }

        .chart-bar-item {
            flex: 1;
            position: relative;
        }

        .chart-bar-fill {
            height: 30px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding: 0 10px;
            color: white;
            font-size: 12px;
            font-weight: bold;
        }

        .chart-config-name {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }
        """

    def _get_scripts(self) -> str:
        """获取 JavaScript 脚本。"""

        return """
        // Tab 切换
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => {
                const targetId = tab.dataset.tab;
                
                // 更新 tab 激活状态
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                
                // 更新内容显示
                document.querySelectorAll('.tab-content').forEach(content => {
                    content.classList.remove('active');
                });
                document.getElementById(targetId).classList.add('active');
            });
        });
        
        // 初始化第一个 tab
        const firstTab = document.querySelector('.tab');
        if (firstTab) {
            firstTab.click();
        }
        """

    def _render_header(
        self,
        report_data: dict[str, Any],
    ) -> str:
        """渲染页面头部。"""

        return f"""
        <div class="header">
            <h1>📊 RAG 评测报告</h1>
            <div class="meta">
                <strong>实验名称：</strong>{report_data['experiment_name']} |
                <strong>实验 ID：</strong>{report_data['experiment_id']} |
                <strong>生成时间：</strong>{report_data['timestamp']}
            </div>
        </div>
        """

    def _render_summary(
        self,
        report_data: dict[str, Any],
    ) -> str:
        """渲染总体摘要。"""

        dataset = report_data["dataset"]
        configs = report_data["configs"]

        total_tests = dataset["size"] * len(configs)
        total_success = sum(
            c["summary"]["success"]
            for c in configs
        )
        total_failure = sum(
            c["summary"]["failure"]
            for c in configs
        )

        success_rate = (
            (total_success / total_tests * 100)
            if total_tests > 0
            else 0
        )

        return f"""
        <div class="section">
            <h2 class="section-title">📈 总体摘要</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">测试配置数</div>
                    <div class="stat-value">{len(configs)}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">测试样本数</div>
                    <div class="stat-value">{dataset['size']}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">成功测试</div>
                    <div class="stat-value" style="color: #27ae60;">{total_success}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">失败测试</div>
                    <div class="stat-value" style="color: #e74c3c;">{total_failure}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">成功率</div>
                    <div class="stat-value" style="color: #667eea;">{success_rate:.1f}%</div>
                </div>
            </div>
        </div>
        """

    def _render_metrics_comparison(
        self,
        report_data: dict[str, Any],
    ) -> str:
        """渲染指标对比。"""

        configs = report_data["configs"]

        metrics_html = ""

        metric_names = [
            ("context_precision", "Context Precision", "检索精度"),
            ("context_recall", "Context Recall", "检索召回"),
            ("faithfulness", "Faithfulness", "忠实度"),
            ("answer_relevancy", "Answer Relevancy", "答案相关性"),
            ("overall_score", "Overall Score", "综合得分"),
        ]

        for metric_key, metric_name_en, metric_name_cn in metric_names:
            chart_bars = ""

            for config in configs:
                score = config["metrics"].get(metric_key, 0)
                width_percent = score * 100

                chart_bars += f"""
                <div class="chart-bar-item">
                    <div class="chart-bar-fill" style="width: {width_percent}%;">
                        {score:.3f}
                    </div>
                    <div class="chart-config-name">{config['name']}</div>
                </div>
                """

            metrics_html += f"""
            <div class="chart-bar">
                <div class="chart-label">{metric_name_cn} ({metric_name_en})</div>
                <div class="chart-bars">
                    {chart_bars}
                </div>
            </div>
            """

        return f"""
        <div class="section">
            <h2 class="section-title">📊 指标对比</h2>
            <div class="comparison-chart">
                {metrics_html}
            </div>
        </div>
        """

    def _render_detailed_results(
        self,
        report_data: dict[str, Any],
    ) -> str:
        """渲染详细结果。"""

        configs = report_data["configs"]

        tabs_html = ""
        contents_html = ""

        for i, config in enumerate(configs):
            active_class = "active" if i == 0 else ""

            tabs_html += f"""
            <button class="tab {active_class}" data-tab="config-{i}">
                {config['name']}
                <span class="badge badge-success">{config['summary']['success']}</span>
                <span class="badge badge-danger">{config['summary']['failure']}</span>
            </button>
            """

            results_html = ""

            for result in config["results"][:50]:  # 限制显示前 50 个
                status = result.get("status", "unknown")

                if status == "success":
                    badge_class = "badge-success"
                    badge_text = "成功"
                elif status == "timeout":
                    badge_class = "badge-warning"
                    badge_text = "超时"
                else:
                    badge_class = "badge-danger"
                    badge_text = "失败"

                metrics = result.get("metrics", {})
                metrics_html = ""

                if metrics:
                    for key, label in [
                        ("context_precision", "检索精度"),
                        ("context_recall", "检索召回"),
                        ("faithfulness", "忠实度"),
                        ("answer_relevancy", "答案相关性"),
                    ]:
                        value = metrics.get(key, 0)
                        metrics_html += f"""
                        <div class="test-metric">
                            <div class="test-metric-label">{label}</div>
                            <div class="test-metric-value">{value:.3f}</div>
                        </div>
                        """

                answer_html = ""

                if result.get("answer"):
                    answer_html = f"""
                    <div class="answer-box">
                        <div class="answer-label">生成答案：</div>
                        {result['answer']}
                    </div>
                    """

                error_html = ""

                if result.get("error"):
                    error_html = f"""
                    <div class="answer-box" style="background: #f8d7da;">
                        <div class="answer-label" style="color: #721c24;">错误信息：</div>
                        {result['error']}
                    </div>
                    """

                results_html += f"""
                <div class="test-result">
                    <div class="test-result-header">
                        <div class="test-query">{result.get('query', 'N/A')}</div>
                        <span class="badge {badge_class}">{badge_text}</span>
                    </div>
                    <div class="test-metrics">
                        {metrics_html}
                    </div>
                    {answer_html}
                    {error_html}
                </div>
                """

            contents_html += f"""
            <div id="config-{i}" class="tab-content {active_class}">
                <div style="margin-bottom: 20px;">
                    <strong>配置详情：</strong>
                    Search Mode: {config['config']['search']['search_mode']}, 
                    Top-K: {config['config']['search']['top_k']}, 
                    Chunk Size: {config['config']['search']['chunk_size']}, 
                    Reflection: {'启用' if config['config']['reflection']['enable_reflection'] else '禁用'}
                </div>
                {results_html}
            </div>
            """

        return f"""
        <div class="section">
            <h2 class="section-title">📝 详细结果</h2>
            <div class="config-tabs">
                {tabs_html}
            </div>
            {contents_html}
        </div>
        """
