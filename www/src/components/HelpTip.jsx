import { useState, useRef, useEffect, useLayoutEffect, useId } from "react";
import { createPortal } from "react-dom";
import HELP_TOPICS from "../help/helpContent.jsx";

const POPOVER_MAX_WIDTH = 340;
const VIEWPORT_MARGIN = 8;

/**
 * A small `?` help icon that toggles an accessible popover with contextual help.
 *
 * Content comes from the {@link HELP_TOPICS} registry via the `topic` prop, or inline via
 * `children`. Interaction: click / Enter / Space toggles; Escape, an outside click, or scrolling
 * dismisses it. The popover is positioned with `position: fixed` and clamped to the viewport, so
 * it never gets clipped by a panel's overflow regardless of where the icon sits.
 *
 * @param {object} props
 * @param {string} [props.topic] Key into the help registry.
 * @param {string} [props.title] Overrides the registry title.
 * @param {string} [props.label] Human label for the aria-label (defaults to the title).
 * @param {import("react").ReactNode} [props.children] Inline content, used instead of `topic`.
 */
export default function HelpTip({ topic, title: titleProp, label, children }) {
  const entry = topic ? HELP_TOPICS[topic] : null;
  const title = titleProp ?? entry?.title ?? "Help";
  const content = children ?? entry?.content ?? null;

  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState({ top: 0, left: 0, width: POPOVER_MAX_WIDTH });
  const btnRef = useRef(null);
  const popRef = useRef(null);
  const popId = useId();

  const place = () => {
    const btn = btnRef.current;
    if (!btn) return;
    const width = Math.min(POPOVER_MAX_WIDTH, window.innerWidth - VIEWPORT_MARGIN * 2);
    const rect = btn.getBoundingClientRect();
    let left = rect.left + rect.width / 2 - width / 2;
    left = Math.max(
      VIEWPORT_MARGIN,
      Math.min(left, window.innerWidth - width - VIEWPORT_MARGIN),
    );
    setPos({ top: rect.bottom + 8, left, width });
  };

  useLayoutEffect(() => {
    if (open) place();
  }, [open]);

  useEffect(() => {
    if (!open) return undefined;
    const onPointerDown = (e) => {
      if (btnRef.current?.contains(e.target) || popRef.current?.contains(e.target)) return;
      setOpen(false);
    };
    const onKeyDown = (e) => {
      if (e.key === "Escape") {
        setOpen(false);
        btnRef.current?.focus();
      }
    };
    const dismiss = () => setOpen(false);
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    // Capture scroll so it fires for any scrolling ancestor, not just the window.
    window.addEventListener("scroll", dismiss, true);
    window.addEventListener("resize", dismiss);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("scroll", dismiss, true);
      window.removeEventListener("resize", dismiss);
    };
  }, [open]);

  if (!content) return null;

  return (
    <span className="help-tip">
      <button
        type="button"
        ref={btnRef}
        className="help-tip-btn"
        aria-label={`Help: ${label ?? title}`}
        aria-expanded={open}
        aria-describedby={open ? popId : undefined}
        onClick={() => setOpen((o) => !o)}
      >
        ?
      </button>
      {open &&
        typeof document !== "undefined" &&
        createPortal(
          <div
            ref={popRef}
            id={popId}
            role="tooltip"
            className="help-pop"
            style={{ top: pos.top, left: pos.left, width: pos.width }}
          >
            <div className="help-pop-title">{title}</div>
            <div className="help-pop-body">{content}</div>
          </div>,
          document.body,
        )}
    </span>
  );
}

/**
 * A form-control label with an optional trailing help icon, kept on one line.
 *
 * Drop-in replacement for a bare `<label htmlFor>` inside a `.control-group`.
 *
 * @param {object} props
 * @param {string} props.htmlFor Id of the control this labels.
 * @param {string} [props.topic] Help topic key; omit for no icon.
 * @param {import("react").ReactNode} props.children Label text.
 */
export function FieldLabel({ htmlFor, topic, children }) {
  return (
    <span className="label-row">
      <label htmlFor={htmlFor}>{children}</label>
      {topic && (
        <HelpTip topic={topic} label={typeof children === "string" ? children : undefined} />
      )}
    </span>
  );
}
