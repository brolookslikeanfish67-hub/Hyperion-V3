"""
Hyperion-V3: Tool Inventory Suite. Provides OS execution sandboxes and native web routing.

PROMPTING RULES FOR MULTI-STEP CRAWLING:
- When given a task, do not guess or rely on your training cutoff weights. 
- You MUST use [BROWSE:url] at least 2-3 times per loop to scrape official data.
- After running [BROWSE:url], feed the raw content text into [EXTRACT_LINKS:url] 
  to discover nested documentation paths, and crawl them sequentially.
"""
import os
import sys
import io
import traceback
import subprocess
import socket
import ssl
import re

class HyperionTools:

    @staticmethod
    def execute_python_code(code: str) -> str:
        """Executes code and gets stdout/stderr."""
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        redirected_out = io.StringIO()
        redirected_err = io.StringIO()
        sys.stdout = redirected_out
        sys.stderr = redirected_err
        try:
            exec(code, {"__builtins__": __builtins__})
            result = redirected_out.getvalue()
        except Exception:
            result = traceback.format_exc()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        return result if result.strip() else "Success (No Output)"

    @staticmethod
    def read_local_file(path: str) -> str:
        """Reads code files for context parsing."""
        if not os.path.exists(path):
            return f"Error: {path} not found."
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return str(e)

    @staticmethod
    def write_local_file(path: str, content: str) -> str:
        """Writes/patches code file matrix lines."""
        try:
            if os.path.dirname(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Successfully written to {path}"
        except Exception as e:
            return str(e)

    @staticmethod
    def run_terminal_command(command: str) -> str:
        """Runs terminal linters or compilers."""
        try:
            res = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=10
            )
            return f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        except Exception as e:
            return str(e)

    @staticmethod
    def raw_http_get(url_path: str, timeout_seconds: float = 5.0) -> str:
        """
        Manually resolves, dials, and scrapes a website at the socket layer.
        Built from scratch with zero third-party dependencies.
        """
        try:
            # 1. Standardize URL and parse Host vs Path
            url_clean = url_path.strip().replace("https://", "").replace("http://", "")
            if "/" in url_clean:
                host, path = url_clean.split("/", 1)
                path = "/" + path
            else:
                host = url_clean
                path = "/"

            print(f"📡 [Socket Engine] Dialing remote host: {host} (Path: {path})")
            
            # 2. Establish raw TCP and wrap with native SSL/TLS
            context = ssl.create_default_context()
            raw_socket = socket.create_connection((host, 443), timeout=timeout_seconds)
            secure_socket = context.wrap_socket(raw_socket, server_hostname=host)

            # 3. Write raw HTTP/1.1 payload from scratch
            http_request = (
                f"GET {path} HTTP/1.1\r\n"
                f"Host: {host}\r\n"
                f"User-Agent: HyperionV3-NativeSocketScraper/1.0\r\n"
                f"Accept: text/html,application/xhtml+xml\r\n"
                f"Connection: close\r\n\r\n"
            )
            secure_socket.sendall(http_request.encode('utf-8'))

            # 4. Stream network buffers back into RAM chunks
            response_bytes = b""
            while True:
                chunk = secure_socket.recv(4096)
                if not chunk:
                    break
                response_bytes += chunk
            secure_socket.close()

            raw_data = response_bytes.decode('utf-8', errors='ignore')
            html_body = raw_data.split("\r\n\r\n", 1)[-1] if "\r\n\r\n" in raw_data else raw_data

            # 5. Regex-based raw structural HTML stripping
            clean_text = re.sub(r"<script.*?>.*?</script>", "", html_body, flags=re.DOTALL)
            clean_text = re.sub(r"<style.*?>.*?</style>", "", clean_text, flags=re.DOTALL)
            clean_text = re.sub(r"<!--.__.?-->", "", clean_text, flags=re.DOTALL)
            clean_text = re.sub(r"<.*?>", " ", clean_text)
            clean_text = re.sub(r"\s+", " ", clean_text).strip()

            # Limit output footprint to keep context window lean for the model
            return f"--- Live Web content for {host}{path} ---\n{clean_text[:2500]}"
        except Exception as e:
            return f"Network Error: Socket scraping failed. Trace: {str(e)}"

    @staticmethod
    def extract_links_from_content(url_context: str, raw_scraped_text: str) -> str:
        """
        Regex engine that parses absolute and relative hyperlinks from text.
        Allows the model to find sub-pages and auto-crawl links recursively.
        """
        try:
            # Isolate the base domain for relative link stitching
            base_url = url_context.strip().replace("https://", "").replace("http://", "").split("/")[0]
            base_prefix = "https://" + base_url

            # Extract matching href patterns
            links = re.findall(r'href=["\'](.*?)["\']', raw_scraped_text)
            
            discovered_links = []
            for link in links:
                link = link.strip()
                if link.startswith("http://") or link.startswith("https://"):
                    discovered_links.append(link)
                elif link.startswith("/") and len(link) > 1:
                    discovered_links.append(f"{base_prefix}{link}")
            
            # Keep the list distinct and slice top 10 choices
            unique_links = list(set(discovered_links))[:10]
            
            if not unique_links:
                return "Crawl Status: No valid outgoing hyper-links detected."
                
            return "--- Discovered Navigation Hyperlinks (Ready to Browse) ---\n" + "\n".join(unique_links)
        except Exception as e:
            return f"Scraper Exception: Link extraction parsing broke. Trace: {str(e)}"
