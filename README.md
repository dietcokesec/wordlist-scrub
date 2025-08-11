# Wordlist Scrubber
This is a super simple tool to scrub non-compliant entries from a wordlist for use in fuzzing/spraying attacks. Nothing here is novel, just neatly packaged. The model of development tries to make it easy to extend for other custom rules you might want.

## Example
It's really easy:
```shell
uv run main.py --min-uppercase-vals 1 --min-lowercase-vals 1 --min-digit-vals 1 --min-val-length 10 rockyou.txt filtered.txt
```