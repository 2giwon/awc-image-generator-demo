import streamlit as st
import numpy as np
import io
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import openai
import logging
import requests

st.set_page_config(page_title="image-generator-demo", page_icon=":shark:", layout="wide")
logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[
        logging.FileHandler("app_errors.log"),  # 파일에 저장
        logging.StreamHandler()  # 터미널에 출력
    ])

def configure_sidebar() -> None:
    
    with st.sidebar:
        st.title("Image Generator Demo")

        if st.secrets["OPEN_AI_KEY"]:
            openai_api_key = st.secrets["OPEN_AI_KEY"]

        if st.secrets["DEEP_AI_KEY"]:
            deep_ai_key = st.secrets["DEEP_AI_KEY"]

        st.subheader("🔑 AI API Key")
        if not openai_api_key:
            openai_api_key = st.text_input("API Key", type="password")
        elif openai_api_key and deep_ai_key:
            st.write("✅ API Key has been set.")
    
        # --- AI 서비스 선택 ---
        st.sidebar.subheader("🔧 Select AI Service")
        ai_service = st.sidebar.radio("Choose AI Provider", ["OpenAI", "DeepAI"], index=0)
        
        if ai_service == "OpenAI":
            with st.expander(":gear: Settings"):
                # 사이드바에서 DALL·E 모델 선택
                st.subheader("🔧 Select DALL·E Model")
                model = st.radio("Choose a model", ["dall-e-2", "dall-e-3"], index=0)

                # 모델별 설정 조정
                if model == "dall-e-3":
                    image_size = "1024x1024"  # DALL·E 3는 1024x1024 고정
                    num_outputs = 1  # DALL·E 3는 한 번에 1개만 생성 가능
                else:
                    image_size = st.radio("Select Image Size", ["256x256", "512x512", "1024x1024"], index=0)
                    num_outputs = st.slider("Number of outputs", min_value=1, max_value=3, value=3)

        elif ai_service == "DeepAI":
            image_size = "512x512"  # DeepAI는 512x512 고정
            num_outputs = 1
            model = "DeepAI"

    return image_size, num_outputs, openai_api_key, deep_ai_key, model

# OpenAI GPT를 사용하여 한글 프롬프트를 최적화된 영어 프롬프트로 변환하는 함수
def optimize_prompt_with_gpt(prompt, api_key):
    try:
        client = openai.OpenAI(api_key=api_key)

        st.write("⏳ Optimizing prompt with GPT...")

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that improves text prompts for AI image generation."},
                {"role": "user", "content": f"Please refine and translate the following text into an optimized English prompt for AI image generation: {prompt}"}
            ],
            temperature=0.7,
            max_tokens=100
        )

        optimized_prompt = response.choices[0].message.content.strip()
        return optimized_prompt

    except Exception as e:
        st.error("❌ Prompt optimization failed.")
        st.error(str(e))
        return prompt  # 실패 시 원본 그대로 사용

def generate_ai_image_from_drawing(image_data, api_key, num_outputs, image_size):
    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API Key!")
        return None

    if image_data is None:
        st.warning("⚠️ Please draw something on the canvas first!")
        return None

    st.write("⏳ Start Processing your drawing with AI...")

    # 이미지 변환 (배경을 흰색으로 설정하고 RGBA 변환)
    image = Image.fromarray((image_data * 255).astype("uint8")).convert("RGBA")

    # 흰색 배경을 가진 새로운 RGBA 이미지 생성
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))  # 완전 불투명한 흰색 배경
    background.paste(image, (0, 0), image)  # 원본 드로잉을 흰색 배경 위에 붙이기

    # 마스크 이미지 생성 (흰색 배경 유지, 검은색 선만 유지)
    mask = image.convert("L")  # 마스크는 그레이스케일 이미지여야 함

    # 이미지 데이터 저장
    img_byte_array = io.BytesIO()
    background.save(img_byte_array, format="PNG")
    img_byte_array.seek(0)  # 버퍼의 처음으로 이동

    mask_byte_array = io.BytesIO()
    mask.save(mask_byte_array, format="PNG")
    mask_byte_array.seek(0)

    st.image(image_data, caption="Your Processed Drawing")

    st.write("⏳ Uploading your drawing to AI...")

    # OpenAI 클라이언트 초기화
    client = openai.OpenAI(api_key=api_key)

    # DALL·E 2의 `edit` 기능 호출
    response = client.images.edit(
        model='dall-e-2',
        image=img_byte_array.getvalue(),  # RGBA 이미지 전달
        mask=mask_byte_array.getvalue(),  # 마스크 이미지 전달
        prompt="Enhance and refine this drawing while maintaining its original structure. Add realistic textures, shading, and colors.",
        n=1,
        size="512x512"
    )

    st.write("✅ AI image generated successfully!")

    return response

def deepai_main_page(deep_ai_key: str, openai_api_key: str) -> None:
    st.title("AI Image Generator Demo")
    option = st.radio("Choose an input method:", ("text to image", "image to image"))

    if option == "text to image":

        st.subheader("📝 Describe the image you want AI to generate")
                
        prompt = st.text_area(
                    ":orange[**Enter prompt: start typing ✍🏾**]",
                    value="A vibrant and dynamic digital illustration of multiple people participating in an online challenge through a mobile app. The participants are actively engaged, checking their rankings on a leaderboard within the app, and striving to improve their performance. The image conveys a sense of competition, motivation, and teamwork. The background features a futuristic interface with charts, rankings, and progress bars, emphasizing the challenge dynamics. The overall atmosphere is energetic and inspiring, with users pushing themselves to achieve higher scores and celebrating their progress together.")

        if st.button("🎨 Generate Image"):
            try:
                if not deep_ai_key:
                    st.warning("⚠️ Please enter your DeepAI API Key!")
                elif not openai_api_key:
                    st.warning("⚠️ Please enter your OpenAI API Key!")
                elif not prompt:
                    st.warning("⚠️ Please enter a description!")
                else:
                    # GPT를 통해 최적화된 프롬프트 변환
                    if prompt:
                        optimized_prompt = optimize_prompt_with_gpt(prompt, openai_api_key)
                        st.write(f"🔄 Optimized Prompt: **{optimized_prompt}**")

                    st.write("⏳ Generating AI image...")
                    r = requests.post(
                        "https://api.deepai.org/api/text2img",
                        data={
                            'text': optimized_prompt,
                        },
                        headers={'api-key': deep_ai_key}
                    )

                    response = r.json()
                    st.write(response)

                    # 이미지 개수에 따라 동적으로 컬럼 생성
                    st.image(response.get("output_url", ""), caption="Generated Image")

            except Exception as e:
                logging.error(f"Error occurred: {e}")
                st.write("⚠️ An unexpected error occurred. Please try again later.")

    elif option == "image to image":
        st.subheader("🖼️ Upload an image to generate a new one")

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        prompt = st.text_area(
                    ":orange[**Enter prompt: start typing ✍🏾**]",
                    value="석양이 지는 노을, 다 같이 뛰는 모습, 앱을 통해 점수를 확인")

        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image")

            if st.button("🎨 Generate Image"):
                try:
                    if not deep_ai_key:
                        st.warning("⚠️ Please enter your DeepAI API Key!")
                    elif not openai_api_key:
                        st.warning("⚠️ Please enter your OpenAI API Key!")
                    elif not prompt:
                        st.warning("⚠️ Please enter a description!")

                    else:
                        # GPT를 통해 최적화된 프롬프트 변환
                        if prompt:
                            optimized_prompt = optimize_prompt_with_gpt(prompt, openai_api_key)
                            st.write(f"🔄 Optimized Prompt: **{optimized_prompt}**")

                        st.write("⏳ Generating AI image...")
                        r = requests.post(
                            "https://api.deepai.org/api/image-editor",
                            files={
                                'image': uploaded_file,
                                'text': optimized_prompt,
                            },
                            headers={'api-key': deep_ai_key}
                        )

                        response = r.json()
                        st.write(response)

                        # 이미지 개수에 따라 동적으로 컬럼 생성
                        st.image(response.get("output_url", ""), caption="Generated Image")

                except Exception as e:
                    logging.error(f"Error occurred: {e}")
                    st.write("⚠️ An unexpected error occurred. Please try again later.")

def openai_main_page(image_size: str , num_outputs: int, openai_api_key: str, deep_ai_key: str, model: str) -> None:
    st.title("AI Image Generator Demo")
    st.subheader("RateLimit 이 걸려있기 때문에 1분에 5개의 요청만 가능합니다.")
    option = st.radio("Choose an input method:", ("Draw an image", "Describe with text"))

    client = openai.OpenAI(api_key=openai_api_key)

    # --- 그림 그리기 기능 ---
    if option == "Draw an image":
        st.subheader("🖌️ Draw your idea on the canvas")

        # 캔버스 생성
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",  # 배경 투명
            stroke_width=5,
            stroke_color="black",
            background_color="white",
            height=300,
            width=500,
            drawing_mode="freedraw",
            key="canvas",
        )

        # AI 이미지 수정 버튼
        if st.button("🎨 Generate Image from Drawing"):
            if canvas_result.image_data is not None:
                response = generate_ai_image_from_drawing(
                    canvas_result.image_data,
                    openai_api_key,
                    num_outputs,
                    image_size,
                )

                if response:
                    # 이미지 개수에 따라 동적으로 컬럼 생성
                    cols = st.columns(1)
                    for idx, img in enumerate(response.data):
                        with cols[idx]:
                            st.image(img.url, caption=f"Generated Image {idx+1}")
                            
            else:
                st.warning("⚠️ Please draw something on the canvas first!")
                

    # --- 텍스트 설명을 통한 이미지 생성 ---
    elif option == "Describe with text":
        st.subheader("📝 Describe the image you want AI to generate")
            
        prompt = st.text_area(
                    ":orange[**Enter prompt: start typing ✍🏾**]",
                    value="A vibrant and dynamic digital illustration of multiple people participating in an online challenge through a mobile app. The participants are actively engaged, checking their rankings on a leaderboard within the app, and striving to improve their performance. The image conveys a sense of competition, motivation, and teamwork. The background features a futuristic interface with charts, rankings, and progress bars, emphasizing the challenge dynamics. The overall atmosphere is energetic and inspiring, with users pushing themselves to achieve higher scores and celebrating their progress together.")  


        if st.button("🎨 Generate Image"):
            try:
                if not openai_api_key:
                    st.warning("⚠️ Please enter your OpenAI API Key!")
                elif not prompt:
                    st.warning("⚠️ Please enter a description!")
                else:
                    # GPT를 통해 최적화된 프롬프트 변환
                    if prompt:
                        optimized_prompt = optimize_prompt_with_gpt(prompt, openai_api_key)
                        st.write(f"🔄 Optimized Prompt: **{optimized_prompt}**")

                    st.write("⏳ Generating AI image...")
                    openai.api_key = openai_api_key
                    response = client.images.generate(
                        model=model,  # 최신 모델 지정
                        prompt=optimized_prompt,
                        n=num_outputs,
                        size=image_size,
                    )
                    
                    # 이미지 개수에 따라 동적으로 컬럼 생성
                    cols = st.columns(num_outputs)
                    for idx, img in enumerate(response.data):
                        with cols[idx]:
                            st.image(img.url, caption=f"Generated Image {idx+1}")

            except Exception as e:
                logging.error(f"Error occurred: {e}")
                st.write("⚠️ An unexpected error occurred. Please try again later.")
                    


def main() -> None:
    image_size, num_outputs, openai_api_key, deep_ai_key, model = configure_sidebar()
    if model == "DeepAI":
        deepai_main_page(deep_ai_key, openai_api_key)
    else:
        openai_main_page(image_size, num_outputs, openai_api_key, deep_ai_key, model)

if __name__ == "__main__":
    main()

