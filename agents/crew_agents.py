"""
AI Agents for Multi-Agent Data Analyst System
==============================================
Uses Groq SDK directly for Python 3.14 compatibility.

Architecture:
  Agent 1 - Data Engineer   -> cleans, profiles data
  Agent 2 - Data Analyst    -> receives Engineer output, runs stats
  Agent 3 - Visualizer      -> receives Analyst output, generates chart code
  Agent 4 - Report Writer   -> receives all prior, writes executive report
  Agent 5 - Quality Gate    -> audits the final report against the data
"""

import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

from agents.prompts import (
    DATA_ENGINEER_PROMPT,
    DATA_ANALYST_PROMPT,
    VISUALIZER_PROMPT,
    REPORT_WRITER_PROMPT,
    QUALITY_GATE_PROMPT,
    DATA_ENGINEER_BACKSTORY,
    DATA_ANALYST_BACKSTORY,
    VISUALIZER_BACKSTORY,
    REPORT_WRITER_BACKSTORY,
    QUALITY_GATE_BACKSTORY,
    DOMAIN_CONTEXTS,
)

load_dotenv()


def _truncate(text: str, max_chars: int = 800) -> str:
    """Return a concise summary of text for passing to next agent."""
    if not text:
        return "(no output)"
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n[truncated - {len(text) - max_chars} more chars]"


def _memory_block(sender: str, content: str, max_chars: int = 800) -> str:
    """Format an agent-to-agent memory block for injection into prompts."""
    return (
        f"--- FROM {sender.upper()} ---\n"
        f"{_truncate(content, max_chars)}\n"
        f"----------------------------"
    )


class GroqAgent:
    """Base agent class using Groq API."""

    def __init__(self, role: str, backstory: str, agent_id: int,
                 model: str = None, temperature: float = 0.1):
        self.role = role
        self.backstory = backstory
        self.agent_id = agent_id
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.temperature = temperature

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        self.client = Groq(api_key=api_key)

    def run(self, task: str) -> str:
        system_prompt = (
            f"You are Agent #{self.agent_id} - {self.role}.\n\n"
            f"{self.backstory}\n\n"
            "Provide detailed, professional responses. Use specific data and numbers when available. "
            "Format your response clearly with sections and bullet points where appropriate."
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": task},
                ],
                temperature=self.temperature,
                max_tokens=1600,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error executing task: {str(e)}"


class DataEngineerAgent(GroqAgent):
    """Agent #1 - Data Engineering specialist."""

    def __init__(self):
        super().__init__(role="Senior Data Engineer", backstory=DATA_ENGINEER_BACKSTORY, agent_id=1)

    def analyze(self, dataset_path: str, data_profile: str, domain_ctx: dict) -> str:
        task = DATA_ENGINEER_PROMPT.format(
            dataset_path=dataset_path,
            domain_framing=domain_ctx["framing"],
            domain_metrics=domain_ctx["priority_metrics"],
        )
        task += f"\n\n## Data Profile\n{data_profile}"
        return self.run(task)


class DataAnalystAgent(GroqAgent):
    """Agent #2 - Data Analysis specialist. Receives Engineer memory."""

    def __init__(self):
        super().__init__(role="Senior Data Analyst", backstory=DATA_ANALYST_BACKSTORY, agent_id=2)

    def analyze(self, data_profile: str, engineering_output: str, domain_ctx: dict) -> str:
        engineering_memory = _memory_block("Data Engineer (Agent #1)", engineering_output)
        task = DATA_ANALYST_PROMPT.format(
            data_profile=data_profile,
            engineering_memory=engineering_memory,
            domain_framing=domain_ctx["framing"],
            domain_lens=domain_ctx["lens"],
            domain_metrics=domain_ctx["priority_metrics"],
        )
        return self.run(task)


class VisualizerAgent(GroqAgent):
    """Agent #3 - Visualisation specialist. Receives Analyst memory."""

    def __init__(self):
        super().__init__(role="Data Visualization Specialist", backstory=VISUALIZER_BACKSTORY, agent_id=3)

    def generate_viz_code(self, analysis_results: str, domain_ctx: dict) -> str:
        task = VISUALIZER_PROMPT.format(analysis_results=analysis_results)
        task = (
            f"Domain: {domain_ctx['framing']} - prioritise charts for: "
            f"{domain_ctx['priority_metrics']}\n\n" + task
        )
        return self.run(task)


class ReportWriterAgent(GroqAgent):
    """Agent #4 - Business Report Writer. Receives full pipeline memory."""

    def __init__(self):
        super().__init__(role="Business Report Writer", backstory=REPORT_WRITER_BACKSTORY, agent_id=4)

    def write_report(self, data_profile: str, analysis_results: str, visualizations: str,
                     engineering_output: str, domain_ctx: dict) -> str:
        task = REPORT_WRITER_PROMPT.format(
            data_profile=data_profile,
            analysis_results=analysis_results,
            visualizations=visualizations,
            domain_framing=domain_ctx["framing"],
            domain_lens=domain_ctx["lens"],
            domain_vocabulary=domain_ctx["vocabulary"],
            engineering_memory=_memory_block("Data Engineer (Agent #1)", engineering_output),
            analysis_memory=_memory_block("Data Analyst (Agent #2)", analysis_results),
        )
        return self.run(task)


class QualityGateAgent(GroqAgent):
    """Agent #5 - Audits final report against actual data for unsupported claims."""

    def __init__(self):
        super().__init__(role="Quality Gate Auditor", backstory=QUALITY_GATE_BACKSTORY,
                         agent_id=5, temperature=0.0)

    def audit(self, report_text: str, data_profile: str,
              analysis_results: str, domain_ctx: dict) -> str:
        task = QUALITY_GATE_PROMPT.format(
            domain=domain_ctx["framing"],
            data_profile=data_profile,
            report_text=report_text,
            analysis_results=_truncate(analysis_results, 1200),
        )
        return self.run(task)


class DataAnalysisCrew:
    """
    Orchestrates 5 AI agents in a memory-linked pipeline.
    """

    def __init__(self, dataset_path: str, domain: str = "General"):
        self.dataset_path = dataset_path
        self.domain = domain
        self.domain_ctx = DOMAIN_CONTEXTS.get(domain, DOMAIN_CONTEXTS["General"])
        self.results: Dict[str, str] = {}
        self.memory_log: List[Dict] = []

        self.data_engineer = DataEngineerAgent()
        self.data_analyst  = DataAnalystAgent()
        self.visualizer    = VisualizerAgent()
        self.report_writer = ReportWriterAgent()
        self.quality_gate  = QualityGateAgent()

    def _log_memory(self, sender: str, receiver: str, content: str, label: str = ""):
        self.memory_log.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "sender": sender,
            "receiver": receiver,
            "label": label,
            "preview": _truncate(content, 400),
            "full": content,
        })

    def run_data_engineering(self, data_profile: str) -> str:
        print("\n#1 Data Engineer working...")
        result = self.data_engineer.analyze(self.dataset_path, data_profile, self.domain_ctx)
        self.results["engineering"] = result
        self._log_memory("Data Engineer (Agent #1)", "Data Analyst (Agent #2)",
                         result, "Cleaned profile + quality report")
        print("   Done")
        return result

    def run_analysis(self, data_profile: str, engineering_output: str) -> str:
        print("\n#2 Data Analyst working...")
        result = self.data_analyst.analyze(data_profile, engineering_output, self.domain_ctx)
        self.results["analysis"] = result
        self._log_memory("Data Analyst (Agent #2)",
                         "Visualizer (Agent #3) + Report Writer (Agent #4)",
                         result, "Statistical findings + key insights")
        print("   Done")
        return result

    def run_visualization(self, analysis_results: str) -> str:
        print("\n#3 Visualizer working...")
        result = self.visualizer.generate_viz_code(analysis_results, self.domain_ctx)
        self.results["visualization"] = result
        self._log_memory("Visualizer (Agent #3)", "Report Writer (Agent #4)",
                         result, "Chart specifications + code")
        print("   Done")
        return result

    def run_report_writing(self, data_profile: str, analysis_results: str,
                            visualizations: str, engineering_output: str) -> str:
        print("\n#4 Report Writer working...")
        result = self.report_writer.write_report(
            data_profile, analysis_results, visualizations, engineering_output, self.domain_ctx)
        self.results["report"] = result
        self._log_memory("Report Writer (Agent #4)", "Quality Gate (Agent #5)",
                         result, "Final executive report")
        print("   Done")
        return result

    def run_quality_gate(self, report_text: str, data_profile: str, analysis_results: str) -> str:
        print("\n#5 Quality Gate Auditor working...")
        result = self.quality_gate.audit(
            report_text, data_profile, analysis_results, self.domain_ctx)
        self.results["quality_gate"] = result
        self._log_memory("Quality Gate (Agent #5)", "USER",
                         result, "Audit verdict + flagged claims")
        print("   Audit complete")
        return result

    def run_full_analysis(self, data_profile: str) -> Dict[str, Any]:
        """Run the complete 5-agent pipeline and return all results + memory log."""
        print("\n" + "=" * 60)
        print(f"MULTI-AGENT PIPELINE  |  Domain: {self.domain}")
        print("=" * 60)

        engineering  = self.run_data_engineering(data_profile)
        analysis     = self.run_analysis(data_profile, engineering)
        viz          = self.run_visualization(analysis)
        report       = self.run_report_writing(data_profile, analysis, viz, engineering)
        quality_gate = self.run_quality_gate(report, data_profile, analysis)

        print("\n" + "=" * 60)
        print("ALL 5 AGENTS COMPLETE")
        print("=" * 60)

        return {
            "engineering":   engineering,
            "analysis":      analysis,
            "visualization": viz,
            "report":        report,
            "quality_gate":  quality_gate,
            "memory_log":    self.memory_log,
            "domain":        self.domain,
        }


def quick_ai_analysis(data_profile: str, dataset_path: str = "dataset",
                       domain: str = "General") -> Dict[str, Any]:
    crew = DataAnalysisCrew(dataset_path, domain=domain)
    return crew.run_full_analysis(data_profile)

