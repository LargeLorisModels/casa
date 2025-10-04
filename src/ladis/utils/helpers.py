import hashlib
from rich.console import Console

def content_hash(*contents: str) -> str:
    """Create a short hash from multiple strings (e.g., grammar + prompt).
    
    Useful for creating unique identifiers for experiments.
    """
    combined = "&?!@&".join(contents)
    return hashlib.md5(combined.encode('utf-8')).hexdigest()[:8]

def get_bar_color(attempts: int) -> str:
    """Determine color based on number of attempts."""
    if attempts > 40:
        return "red"
    elif attempts > 15:
        return "yellow"
    else:
        return "green"
    
def print_progress(sample_idx: int, n_samples: int, attempts: int, max_attempts: int, verbose: bool, timeout: bool):
    """Print progress bar for current sample."""
    if not verbose:
        return
    
    console = Console()
    if not timeout:
        color = get_bar_color(attempts)
        
        # Scale bar proportionally to max attempts seen
        bar_length = max(1, int((attempts / max_attempts) * 40))
        bar = "█" * bar_length
        
        console.print(
            f"Sample {sample_idx:02d}/{n_samples}: [{color}]{bar}[/{color}] {attempts} attempts"
        )
    else:
        console.print(f"Sample {sample_idx:02d}/{n_samples}: [red]{'█' * 40} {max_attempts} attempts (timeout)[/red]")
        