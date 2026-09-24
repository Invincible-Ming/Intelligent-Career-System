export default function (component) {
    const {data, parentElement} = component;
    const form = parentElement.querySelector("form");
    const status = parentElement.querySelector(".status");
    const retry = parentElement.querySelector(".retry");
    const button = form.querySelector("button");
    const controller = new AbortController();
    let active = true;
    const loggingOut = data.mode === "logout";
    form.hidden = loggingOut;

    async function submit() {
        button.disabled = true;
        retry.hidden = true;
        status.hidden = false;
        status.classList.remove("error");
        status.textContent = loggingOut ? "正在退出登录…" : "正在登录…";
        const body = new URLSearchParams({return_to: data.return_to});
        if (!loggingOut) {
            body.set("username", form.elements.username.value);
            body.set("password", form.elements.password.value);
        }
        try {
            // The response never contains the token. The browser stores Set-Cookie.
            const response = await fetch(data.url, {
                method: "POST", credentials: "include", cache: "no-store", signal: controller.signal,
                headers: {Accept: "application/json", "Content-Type": "application/x-www-form-urlencoded"},
                body
            });
            const result = await response.json();
            if (!active) return;
            if (!response.ok || !(loggingOut ? result.cleared : result.logged_in)) {
                throw new Error(typeof result.detail === "string" ? result.detail : "请检查输入后重试");
            }
            if (!loggingOut) form.elements.password.value = "";
            // A new WebSocket request allows Streamlit to receive the HttpOnly cookie.
            window.location.reload();
        } catch (error) {
            if (!active) return;
            status.classList.add("error");
            status.textContent = error instanceof TypeError ? "无法连接后端，请检查连接后重试。" : error.message;
            retry.hidden = !loggingOut;
            button.disabled = false;
        }
    }

    const onSubmit = event => {
        event.preventDefault();
        submit();
    };
    form.addEventListener("submit", onSubmit);
    retry.addEventListener("click", submit);
    if (loggingOut) submit();
    return () => {
        active = false;
        controller.abort();
        form.removeEventListener("submit", onSubmit);
        retry.removeEventListener("click", submit);
    };
}
