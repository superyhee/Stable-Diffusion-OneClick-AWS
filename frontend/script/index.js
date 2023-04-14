const TYPE_GENARATE = {
  txt2img: "txt2img",
  img2img: "img2img",
};
const inputHeight = document.getElementById("input-height");
const labelInputHeight = document.querySelector(".input__height--value");
const inputWidth = document.getElementById("input-width");
const labelInputWidth = document.querySelector(".input__width--value");
const textInput = document.getElementById("text-input");
const urlInput = document.getElementById("url-input");
const imageGenerated = document.getElementById("image-generated");
const imageLoading = document.querySelector(".image-loading");
const BULLET_WIDTH = 16;
const PADDING_CONTAINER = 24;

function getDataText2Image() {
  const prompt = textInput === null ? "" : textInput.value;
  const width = inputWidth === null ? 512 : inputWidth.valueAsNumber;
  const height = inputHeight === null ? 512 : inputHeight.valueAsNumber;
  return {
    type: TYPE_GENARATE.txt2img,
    prompt,
    height,
    width,
  };
}

function getDataImage2Image() {
  const prompt = textInput === null ? "" : textInput.value;
  const url = urlInput === null ? "" : urlInput.value;
  return {
    type: TYPE_GENARATE.txt2img,
    prompt,
    url,
  };
}

function postData(url = "", data = {}) {
  return fetch(url, {
    method: "POST",
    mode: "cors",
    cache: "no-cache",
    credentials: "same-origin",
    headers: {
      "Content-Type": "application/json",
    },
    redirect: "follow",
    referrerPolicy: "no-referrer",

    body: JSON.stringify(data),
  }).then((res) => {
    return res.blob();
  });
}

function loadingImage(isLoading) {
  if (isLoading) {
    imageLoading.classList.remove("d-none");
  } else {
    imageLoading.classList.add("d-none");
  }
}

function onClickBtnGenerate(type, params) {
  const isType = Object.keys(TYPE_GENARATE).some(
    (typeDefault) => typeDefault === type
  );
  if (!isType) return;
  loadingImage(true);
  postData(`${window.location.origin}/${type}`, params).then((file) => {
    loadingImage(false);
    const urlImageGenerate = URL.createObjectURL(file);
    imageGenerated.src = urlImageGenerate;
  }).catch((error) => {
    loadingImage(false);
    console.error("error:", error)
  });
}

document.getElementById("btn-submit").addEventListener("click", function () {
  const type = document.querySelector(".nav-link.active").id;
  if (type === TYPE_GENARATE.txt2img) {
    const params = getDataText2Image();
    onClickBtnGenerate(TYPE_GENARATE.txt2img, params);
  } else {
    const params = getDataImage2Image();
    onClickBtnGenerate(TYPE_GENARATE.img2img, params);
  }
});

function calcPositionLabel(input, label) {
  label.innerHTML = input.value;
  let bulletPosition = (input.value - input.min) / (input.max - input.min);
  label.style.left =
    bulletPosition * (input.offsetWidth - BULLET_WIDTH) +
    PADDING_CONTAINER +
    "px";
  if (input.valueAsNumber > 100) {
    label.style.marginLeft = "-5px";
  } else {
    label.style.marginLeft = 0;
  }
}

inputHeight.addEventListener("input", function () {
  calcPositionLabel(inputHeight, labelInputHeight);
});

inputWidth.addEventListener("input", function () {
  calcPositionLabel(inputWidth, labelInputWidth);
});

window.addEventListener("load", function () {
  calcPositionLabel(inputHeight, labelInputHeight);
  calcPositionLabel(inputWidth, labelInputWidth);
});

document.querySelectorAll(".nav-link").forEach((navLink) =>
  navLink.addEventListener("click", function (e) {
    document.querySelector(".nav-link.active").classList.remove("active");
    e.target.classList.add("active");
    imageGenerated.src = "";
    if (e.target.id === TYPE_GENARATE.img2img) {
      document.querySelector(".url-textarea").classList.remove("d-none");
      document.querySelector(".image-resizer").classList.add("d-none");
    } else {
      document.querySelector(".image-resizer").classList.remove("d-none");
      document.querySelector(".url-textarea").classList.add("d-none");
    }
  })
);
