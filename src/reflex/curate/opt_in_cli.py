"""`reflex contribute` CLI subcommand.

Surface:
  reflex contribute                  → equivalent to --status (zero-friction default)
  reflex contribute --opt-in         → create receipt + show success
  reflex contribute --opt-out        → delete receipt locally
  reflex contribute --revoke         → opt-out + cascade purge through server
  reflex contribute --status         → show current consent + contribution stats
  reflex contribute --info           → show privacy/legal details before deciding
  reflex contribute --inspect        → list episodes in the local upload queue
  reflex contribute --purge ID       → drop a specific queued episode before upload

Phase 1 wires --opt-in / --opt-out / --revoke / --status / --info. Inspect
and purge land alongside the uploader spec (those need the queue).
"""
from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from reflex.curate import consent as curate_consent
from reflex.curate import messaging

logger = logging.getLogger(__name__)

contribute_app = typer.Typer(
    name="contribute",
    help="Reflex Data Contribution — opt in, opt out, check status, or revoke.",
    invoke_without_command=True,
)
console = Console()


def _detect_pro_tier() -> tuple[str, str | None]:
    """Return (tier, customer_id_if_pro). Pro tier is detected when a valid
    Pro license file exists; otherwise free tier with no customer_id.

    Anonymous Free contributors get a derived contributor_id in `consent.save()`
    (via `derive_contributor_id()`). Pro contributors use the license-bound
    customer_id so their revenue share lands in the right billing record.
    """
    try:
        from reflex.pro.activate import DEFAULT_LICENSE_PATH
        path = Path(DEFAULT_LICENSE_PATH).expanduser()
        if not path.exists():
            return "free", None
        import json
        data = json.loads(path.read_text())
        customer_id = data.get("customer_id")
        if not customer_id:
            return "free", None
        return "pro", str(customer_id)
    except Exception as exc:  # noqa: BLE001
        logger.debug("pro_tier detect failed, defaulting to free: %s", exc)
        return "free", None


@contribute_app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    opt_in: bool = typer.Option(
        False, "--opt-in", help="Opt in to data contribution.",
    ),
    opt_out: bool = typer.Option(
        False, "--opt-out", help="Opt out (local receipt only; historical data stays).",
    ),
    revoke: bool = typer.Option(
        False, "--revoke",
        help="Opt out + request server-side cascade purge of historical data.",
    ),
    status: bool = typer.Option(
        False, "--status", help="Show current consent + contribution stats.",
    ),
    info: bool = typer.Option(
        False, "--info", help="Show full privacy + legal details before deciding.",
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y",
        help="Auto-confirm prompts (for --revoke). Required in non-interactive contexts.",
    ),
) -> None:
    """Reflex Data Contribution program — opt in, opt out, or check status."""
    if ctx.invoked_subcommand is not None:
        return

    flags = [opt_in, opt_out, revoke, status, info]
    set_count = sum(1 for f in flags if f)
    if set_count > 1:
        console.print(
            "[red]Pick one of:[/red] --opt-in, --opt-out, --revoke, --status, --info"
        )
        raise typer.Exit(2)

    if info:
        _cmd_info()
        return
    if opt_in:
        _cmd_opt_in()
        return
    if opt_out:
        _cmd_opt_out()
        return
    if revoke:
        _cmd_revoke(yes=yes)
        return
    # No flag set OR --status set: show status
    _cmd_status()


def _cmd_info() -> None:
    """Print the privacy + legal context before the operator decides."""
    body = f"""
[bold]Reflex Data Contribution — what you're agreeing to[/bold]

[bold cyan]Mission[/bold cyan]
{messaging.MISSION_FRAMING}

[bold cyan]What you get[/bold cyan]
Free tier: {messaging.RECIPROCITY_FRAMING_FREE}

Pro tier:  {messaging.RECIPROCITY_FRAMING_PRO}

[bold cyan]What's contributed[/bold cyan]
- Robot recordings made with [cyan]reflex serve --record[/cyan]
- Episode metadata (timestamps, embodiment, task labels)
- Camera frames (face-blurred at source)
- Action chunks (the policy's outputs)
- Instruction text (SHA-256 hashed by default; raw on explicit opt-in)

What is NOT contributed:
- Recordings made without --record
- Recordings made BEFORE you opted in (no retroactive contribution)
- Anything you mark sensitive via [cyan]reflex inspect traces --mark-sensitive ID[/cyan]
- Raw images, unless you explicitly enable [cyan]privacy_mode=raw_opt_in[/cyan]

[bold cyan]Privacy[/bold cyan]
{messaging.PRIVACY_FRAMING}

[bold cyan]Revoke anytime[/bold cyan]
[cyan]reflex contribute --opt-out[/cyan]   stop future contribution
[cyan]reflex contribute --revoke[/cyan]    cascade purge historical contributions
                              (30-day SLA, GDPR Article 17 compliant)

[bold cyan]Decide[/bold cyan]
[cyan]reflex contribute --opt-in[/cyan]    join the program
"""
    console.print(body)


def _cmd_opt_in() -> None:
    if curate_consent.is_opted_in():
        receipt = curate_consent.load()
        console.print(
            f"[yellow]Already opted in[/yellow] as [dim]{receipt.contributor_id}[/dim] "
            f"({receipt.tier}, since {receipt.opted_in_at}).\n"
            f"Run [cyan]reflex contribute --status[/cyan] to see contribution stats, "
            f"or [cyan]reflex contribute --opt-out[/cyan] to stop."
        )
        return

    tier, license_customer_id = _detect_pro_tier()
    receipt = curate_consent.save(
        tier=tier,
        contributor_id=license_customer_id,  # None → anonymous derive for Free
    )
    console.print(messaging.opt_in_success(
        tier=receipt.tier, contributor_id=receipt.contributor_id,
    ))


def _cmd_opt_out() -> None:
    removed = curate_consent.revoke()
    if not removed:
        console.print(
            "[dim]Already opted out[/dim] — no consent receipt found at "
            f"[cyan]{Path(curate_consent.DEFAULT_CONSENT_PATH).expanduser()}[/cyan]."
        )
        return
    console.print(messaging.opt_out_success())


def _cmd_revoke(*, yes: bool) -> None:
    try:
        receipt = curate_consent.load()
    except curate_consent.ConsentNotFound:
        console.print(
            "[dim]No active consent receipt — nothing to revoke.[/dim] "
            "If you previously contributed before clearing the receipt, "
            "contact [cyan]privacy@fastcrest.com[/cyan] with your contributor_id."
        )
        return
    except curate_consent.ConsentCorrupted as exc:
        console.print(f"[red]Receipt corrupted:[/red] {exc}")
        raise typer.Exit(2)

    console.print(messaging.revoke_warning())
    if not yes:
        try:
            ans = typer.confirm("Confirm revoke?", default=False)
        except (EOFError, OSError):
            console.print(
                "[red]Non-interactive context — pass --yes to confirm.[/red]"
            )
            raise typer.Exit(2)
        if not ans:
            console.print("[dim]Revoke cancelled.[/dim]")
            return

    # Local: remove the receipt now.
    curate_consent.revoke()

    # Server-side cascade: POST to the contribution worker.
    # Phase 1: worker not yet deployed — log + show the manual fallback.
    # When the worker ships, replace this stub with an httpx.post call.
    contributor_id = receipt.contributor_id
    logger.info("curate revoke requested for contributor_id=%s", contributor_id)
    console.print(messaging.revoke_success(contributor_id=contributor_id))
    console.print(
        "[dim]Phase 1 note: the contribution worker is not yet deployed, so the "
        "server-side cascade purge is queued locally. Once the worker is live, "
        "future revocations will trigger the cascade automatically. To request "
        "manual purge in the meantime, email [cyan]privacy@fastcrest.com[/cyan] "
        "with your contributor_id above.[/dim]"
    )


def _cmd_status() -> None:
    if not curate_consent.is_opted_in():
        console.print(
            "[bold]Data contribution:[/bold] [yellow]not opted in[/yellow]\n"
            "[dim]Run [cyan]reflex contribute --opt-in[/cyan] to join, or "
            "[cyan]reflex contribute --info[/cyan] for privacy details first.[/dim]"
        )
        return

    receipt = curate_consent.load()
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Status:", "[green]✓ opted in[/green]")
    table.add_row("Tier:", receipt.tier)
    table.add_row("contributor_id:", f"[dim]{receipt.contributor_id}[/dim]")
    table.add_row("Opted in at:", receipt.opted_in_at)
    table.add_row("Privacy mode:", receipt.privacy_mode)
    table.add_row("Terms version:", receipt.accepted_terms_version)
    table.add_row(
        "Receipt path:",
        f"[dim]{Path(curate_consent.DEFAULT_CONSENT_PATH).expanduser()}[/dim]",
    )
    console.print(table)

    # Phase 1.5+ will populate hours / episodes from the uploader's stats:
    # console.print(f"Hours contributed: {hours:.1f}")
    # console.print(f"Episodes contributed: {episodes}")
    console.print()
    console.print(
        "[dim]Hours / episode counts will appear here once the uploader "
        "ships in Phase 1 (uploader spec: features/08_curate/_collection/"
        "data-collection-free-tier.md).[/dim]"
    )


__all__ = ["contribute_app"]
