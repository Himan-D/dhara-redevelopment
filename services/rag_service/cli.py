#!/usr/bin/env python3
"""
Unified RAG + Property Card Analysis CLI
Combines DCPR 2034 knowledge base with property card workflow
"""

import argparse
import sys
from pathlib import Path

# Ensure the repository root is on Python path when running cli.py directly.
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def main():
    parser = argparse.ArgumentParser(
        description="DCPR RAG + Property Card Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query DCPR regulations
  python3 cli.py query "What is FSI for residential buildings?"
  
  # Analyze property card and generate reports
  python3 cli.py analyze --survey-no "123/P" --area 2200 --road-width 12
  
  # Process existing property card image/PDF
  python3 cli.py scan --input property_card.pdf --output reports/
  
  # Generate scheme comparison
  python3 cli.py compare --area 2200 --schemes 33(7B) 33(20B) 30(A)
  
  # Interactive mode
  python3 cli.py interactive
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query DCPR regulations")
    query_parser.add_argument("question", help="Your question")
    query_parser.add_argument("--model", default="qwen2.5:7b", help="LLM model")
    query_parser.add_argument("--k", type=int, default=5, help="Number of results")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze property")
    analyze_parser.add_argument("--survey-no", required=True, help="Survey number")
    analyze_parser.add_argument(
        "--area", type=float, required=True, help="Plot area in sq.m"
    )
    analyze_parser.add_argument(
        "--road-width", type=float, default=9, help="Road width in meters"
    )
    analyze_parser.add_argument(
        "--zone",
        default="Residential",
        choices=["Residential", "Commercial", "Industrial"],
    )
    analyze_parser.add_argument("--village", default="", help="Village name")
    analyze_parser.add_argument("--taluka", default="", help="Taluka name")
    analyze_parser.add_argument("--district", default="Mumbai", help="District name")
    analyze_parser.add_argument("--scheme", default="33(7B)", help="DCPR scheme to use")
    analyze_parser.add_argument(
        "--affordable-housing",
        type=float,
        default=70,
        help="Affordable housing percentage for 33(7B) incentive (default: 70)",
    )
    analyze_parser.add_argument(
        "--residential-rate",
        type=float,
        default=50000,
        help="Rate per sq.ft for residential",
    )
    analyze_parser.add_argument("--output", default="reports/", help="Output directory")

    # Scan command
    scan_parser = subparsers.add_parser(
        "scan", help="Scan property card from image/PDF"
    )
    scan_parser.add_argument("--input", required=True, help="Input file (PDF or image)")
    scan_parser.add_argument("--output", default="reports/", help="Output directory")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare DCPR schemes")
    compare_parser.add_argument(
        "--area", type=float, required=True, help="Plot area in sq.m"
    )
    compare_parser.add_argument(
        "--schemes",
        nargs="+",
        default=["33(20B)", "33(11)", "33(7B)", "30(A)"],
        help="Schemes to compare",
    )
    compare_parser.add_argument(
        "--affordable-housing",
        type=float,
        default=70,
        help="Affordable housing percentage for 33(7B) incentive (default: 70)",
    )

    # Interactive command
    subparsers.add_parser("interactive", help="Start interactive mode")

    # Stats command
    subparsers.add_parser("stats", help="Show system statistics")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index DCPR document")
    index_parser.add_argument("--pdf", help="PDF to index (single file)")
    index_parser.add_argument("--rebuild", action="store_true", help="Rebuild index")
    index_parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Run full pipeline on all docs in data/docs (extract, clean, chunk, embed)",
    )
    index_parser.add_argument(
        "--workers", type=int, default=4, help="Extraction workers for pipeline"
    )
    index_parser.add_argument(
        "--resume", action="store_true", help="Resume pipeline from last progress"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Handle commands
    if args.command == "query":
        handle_query(args)
    elif args.command == "analyze":
        handle_analyze(args)
    elif args.command == "scan":
        handle_scan(args)
    elif args.command == "compare":
        handle_compare(args)
    elif args.command == "interactive":
        handle_interactive(args)
    elif args.command == "stats":
        handle_stats(args)
    elif args.command == "index":
        handle_index(args)


def handle_query(args):
    """Query DCPR regulations - using Intelligent RAG Agent"""
    import os
    from services.rag_service.services import IntelligentRAG
    from pathlib import Path

    # Load API key from .env file
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().strip().split("\n"):
            if "=" in line:
                key, val = line.split("=", 1)
                os.environ[key] = val

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found. Set it in .env file.")
        return

    print("Querying DCPR regulations (Intelligent RAG)...\n")

    # Use Intelligent RAG
    agent = IntelligentRAG()
    result = agent.query(args.question)

    # Print answer
    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(result["answer"])
    print("\n" + "=" * 80)

    if result.get("sources"):
        print("SOURCES")
        print("-" * 20)
        for i, source in enumerate(result["sources"][:3], 1):
            text = source.get("text", "")[:200].replace("\n", " ")
            print(f"[{i}] {text}...")


def handle_analyze(args):
    """Analyze property and generate reports"""
    from services.rag_service.services.property_card_workflow import (
        PropertyCardWorkflow,
        PropertyCard,
        RevenueBreakdown,
    )

    print(f"Analyzing property: {args.survey_no}")

    # Sanitize survey number for file names
    safe_survey_no = args.survey_no.replace("/", "_").replace("\\", "_")

    # Create property card
    card = PropertyCard(
        survey_no=args.survey_no,
        plot_area_sq_m=args.area,
        plot_area_sq_ft=args.area * 10.764,
        road_width_m=args.road_width,
        zone_type=args.zone,
        village=args.village,
        taluka=args.taluka,
        district=args.district,
    )

    # Create revenue model
    revenue = RevenueBreakdown(
        residential_area_sqft=card.plot_area_sq_ft * 2.5 * 0.8,  # 2.5 FSI, 80% saleable
        residential_rate_per_sqft=args.residential_rate,
        parking_slots=int(card.plot_area_sq_ft / 500),
    )

    # Initialize workflow
    workflow = PropertyCardWorkflow()

    # Analyze
    analysis = workflow.analyze_from_card(
        card,
        schemes=[args.scheme],
        revenue=revenue,
        affordable_housing_pct=args.affordable_housing,
    )
    analysis.project_name = f"Property_{safe_survey_no}"

    # Generate reports
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating reports...")

    # Scheme comparison (text)
    scheme_file = output_dir / f"{safe_survey_no}_scheme_comparison.txt"
    with open(scheme_file, "w", encoding="utf-8") as f:
        f.write(
            workflow.generator.generate_scheme_comparison(
                analysis, ["33(20B)", "33(11)", "33(7B)", "30(A)"]
            )
        )
    print(f"  ✓ Scheme comparison: {scheme_file}")

    # Financial summary (PDF)
    financial_file = output_dir / f"{safe_survey_no}_financial_summary.pdf"
    workflow.generator.export_to_pdf(analysis, str(financial_file), "financial")
    print(f"  ✓ Financial summary: {financial_file}")

    # Approval cost summary (text)
    approval_file = output_dir / f"{safe_survey_no}_approval_costs.txt"
    with open(approval_file, "w", encoding="utf-8") as f:
        f.write(workflow.generator.generate_approval_cost_summary(analysis))
    print(f"  ✓ Approval costs: {approval_file}")

    print(f"\nReports generated in: {output_dir}")


def handle_scan(args):
    """Scan property card from file"""
    from services.rag_service.services.property_card_workflow import PropertyCardWorkflow

    print(f"Scanning: {args.input}")
    workflow = PropertyCardWorkflow()

    try:
        outputs = workflow.run_workflow(args.input, args.output)
        print("\nReports generated:")
        for report_type, path in outputs.items():
            print(f"  {report_type}: {path}")
    except Exception as e:
        print(f"Error scanning: {e}")


def handle_compare(args):
    """Compare DCPR schemes"""
    from services.rag_service.services.property_card_workflow import DCPRCalculator, PropertyCard

    print(f"Comparing schemes for {args.area} sq.m plot\n")

    calculator = DCPRCalculator()
    card = PropertyCard(plot_area_sq_m=args.area)

    print(
        f"{'Scheme':<15} {'Basic FSI':<12} {'Incentive':<12} {'Max FSI':<12} {'Premium':<12}"
    )
    print("-" * 65)

    for scheme in args.schemes:
        try:
            config = calculator.calculate_scheme(
                scheme,
                card.plot_area_sq_m,
                road_width_m=12,
                zone_type="Residential",
                affordable_housing_pct=args.affordable_housing
                if scheme == "33(7B)"
                else 0,
            )
            print(
                f"{scheme:<15} {config.basic_fsi:<12.2f} {config.incentive_fsi:<12.2f} "
                f"{config.max_permissible_fsi:<12.2f} {config.premium_fsi:<12.2f}"
            )
        except Exception as e:
            print(f"{scheme:<15} Error: {e}")

    print()


def handle_interactive(args):
    """Interactive mode"""
    print("=" * 60)
    print("DCPR RAG + Property Card Analysis System")
    print("=" * 60)
    print("\nCommands:")
    print("  query <question>  - Query DCPR regulations")
    print("  analyze <params>  - Analyze a property")
    print("  compare <area>   - Compare schemes")
    print("  help              - Show help")
    print("  exit              - Exit\n")

    while True:
        try:
            user_input = input("> ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break

            if user_input.lower() in ["help", "?"]:
                print("Available commands:")
                print("  query <question>  - Query DCPR regulations")
                print("  analyze <params>  - Analyze a property")
                print("  compare <area>   - Compare schemes")
                print("  help              - Show this help")
                print("  exit              - Exit")
                continue

            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            rest = parts[1] if len(parts) > 1 else ""

            if cmd == "query" and rest:
                args.question = rest
                handle_query(args)
            elif cmd == "compare" and rest:
                args.area = float(rest)
                args.schemes = ["33(20B)", "33(11)", "33(7B)", "30(A)"]
                handle_compare(args)
            elif cmd == "analyze" and rest:
                print("Use: analyze --survey-no 123/P --area 2200")
            else:
                print(f"Unknown command: {cmd}. Type 'help' for available commands.")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def handle_stats(args):
    """Show system statistics"""
    from pymilvus import connections, utility

    print("=" * 60)
    print("SYSTEM STATISTICS")
    print("=" * 60)

    # Milvus stats
    print("\nVector Database (Milvus):")
    try:
        connections.connect("default", host="localhost", port="19530")
        collections = utility.list_collections()
        print(f"  Collections: {len(collections)}")
        for coll in collections:
            from pymilvus import Collection

            c = Collection(coll)
            print(f"    - {coll}: {c.num_entities} entities")
        connections.disconnect("default")
    except Exception as e:
        print(f"  Error: {e}")

    # File stats
    print("\nData Files:")
    data_dir = Path("data")
    if data_dir.exists():
        vectors_dir = data_dir / "vectors"
        if vectors_dir.exists():
            files = list(vectors_dir.glob("*.json"))
            print(f"  Vector files: {len(files)}")
            for f in files:
                print(f"    - {f.name}")

    # Feedback stats
    feedback_file = data_dir / "feedback.json"
    if feedback_file.exists():
        import json

        feedback = json.loads(feedback_file.read_text())
        print(f"  Feedback entries: {len(feedback)}")

    print()


def handle_index(args):
    """Index DCPR documents"""
    import os

    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().strip().split("\n"):
            if "=" in line:
                key, val = line.split("=", 1)
                os.environ[key] = val

    # Full pipeline mode: process all docs in data/docs
    if args.pipeline:
        print("Running full index pipeline on data/docs...")
        from scripts.index_pipeline import run_pipeline

        run_pipeline(
            docs_dir="data/docs",
            drop_existing=args.rebuild,
            max_extract_workers=args.workers,
            resume=args.resume,
        )
        return

    # Single file mode (legacy)
    print("Indexing single document...")

    if not args.pdf:
        pdf_path = Path("/home/ubuntu/DCPR 2034 updated upto 12092024.pdf")
    else:
        pdf_path = Path(args.pdf)

    if not pdf_path.exists():
        if not pdf_path.is_absolute():
            potential_path = Path(os.getcwd()) / args.pdf
            if potential_path.exists():
                pdf_path = potential_path
            else:
                print(f"File not found: {pdf_path}")
                return
        else:
            print(f"File not found: {pdf_path}")
            return

    from rag import RAGAgent, DocumentLoader
    from pymilvus import connections, utility

    # Clear existing collection if rebuild
    if args.rebuild:
        print("Rebuilding index...")
        connections.connect("default", host="localhost", port="19530")
        if utility.has_collection("documents"):
            utility.drop_collection("documents")
            print("  Dropped existing collection")
        connections.disconnect("default")

    # Load and index
    print(f"Loading: {pdf_path}")
    if pdf_path.suffix.lower() == ".pdf":
        text = DocumentLoader.load_pdf(pdf_path)
    elif pdf_path.suffix.lower() == ".txt":
        text = DocumentLoader.load_text(pdf_path)
    elif pdf_path.suffix.lower() == ".docx":
        text = DocumentLoader.load_docx(pdf_path)
    else:
        print(f"Unsupported file type: {pdf_path.suffix}")
        return

    chunks = DocumentLoader.chunk_text(text)
    print(f"Created {len(chunks)} chunks")

    agent = RAGAgent(use_milvus=True)

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        agent.vectorstore.add_documents(batch)
        print(f"  Indexed {min(i + batch_size, len(chunks))}/{len(chunks)}")

    print("Indexing complete!")


if __name__ == "__main__":
    main()
