"""Describe the purpose of this script here..."""
from codetiming import Timer
from humanfriendly import format_timespan


@Timer(text=lambda secs: f"Took {format_timespan(secs)}")
def main():
    pass
    

if __name__ == "__main__":
    main()
