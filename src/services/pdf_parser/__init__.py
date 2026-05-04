from .factory import get_pdf_parser_client


def parse_pdf(content: bytes) -> str:
    return get_pdf_parser_client().parse_pdf(content)
