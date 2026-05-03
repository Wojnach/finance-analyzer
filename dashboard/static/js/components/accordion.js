/*
 * accordion.js — collapsible card.
 *
 * The body is appended once and toggled via a class. Caller passes the
 * pre-rendered body as a Node; this component does not parse HTML strings.
 */

/**
 * @param {{title: string, body: Node, openByDefault?: boolean}} props
 * @returns {HTMLElement}
 */
export function accordion({ title, body, openByDefault = false } = {}) {
  const root = document.createElement("section");
  root.className = "accordion" + (openByDefault ? " open" : "");

  const head = document.createElement("button");
  head.type = "button";
  head.className = "accordion__head";
  head.setAttribute("aria-expanded", String(openByDefault));

  const titleEl = document.createElement("span");
  titleEl.textContent = title || "";
  head.append(titleEl);

  const chev = document.createElement("span");
  chev.className = "accordion__chevron";
  chev.setAttribute("aria-hidden", "true");
  chev.textContent = "▾";
  head.append(chev);

  const bodyWrap = document.createElement("div");
  bodyWrap.className = "accordion__body";
  if (body) bodyWrap.append(body);

  head.addEventListener("click", () => {
    const open = !root.classList.contains("open");
    root.classList.toggle("open", open);
    head.setAttribute("aria-expanded", String(open));
  });

  root.append(head, bodyWrap);
  return root;
}
