
import random
import time
import torch
import subprocess
import json
import os
import signal
import pickle
import csv
import whisper
import speech_recognition as sr
# import rag_viper_sample_code.py_rag_offloader as pro
import streamlit as st
import numpy as np


from streamlit_lottie import st_lottie
from streamlit import _bottom
from typing import Any
from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
from tqdm import tqdm


path = "./parsing_data"
# embedding_model = SentenceTransformer("WhereIsAI/UAE-Large-V1")

def example_generator():

    example_list =[]
    # st.seesion_state.model_name, st.session_state.rag_on, st.session_state.rag_db

    if st.session_state.model_name == "TAIDE":
        if st.session_state.rag_on:
            if st.session_state.rag_db == "DBpedia":
                example_list =["Tell me the history of Super Mario",
                               "Tell me the history of computex",
                               "Give me a brief introduction about Apple Inc.?"]
        else:
            example_list=["幫我規劃台北一日遊的行程",
                          "請幫我總結台版晶片法案",
                          "台灣保育類物種"
                            ]
    else:
        if st.session_state.rag_on:
            if st.session_state.rag_db == "DBpedia":
                example_list =["Tell me the history of Super Mario",
                               "Tell me the history of computex",
                               "Give me a brief introduction about Apple Inc.?"]
        else:
            example_list= ["Safety Precautions for mountain climbing",
                           " Python codes for finding the prime",
                           " Tell me an interesting fact about Egypt"]
            
    return example_list

def rag_db_prepare():

    return 0

def rag_process(prompt, corpus_emb, raw_text, model):

    embedding_model = model
    # global rag_offloader
    prompt_emb = embedding_model.encode(prompt, convert_to_tensor=True)

    if st.session_state.rag_device == "viper":


        # prompt_emb_arr = torch.squeeze(prompt_emb, 0).numpy().astype(np.int8)
        scale = 0.0374
        prompt_emb = torch.clamp(torch.round(prompt_emb/scale), -127, 127).to(torch.int8)
        prompt_emb_arr = prompt_emb.view(-1, 16).flip(1).flatten().numpy()
        
        output_scores = np.zeros((4001792,), dtype=np.int8)
        print("output scores shape: ", output_scores.shape)

        # dot product on viper
        st.session_state.rag_offloader.run(prompt_emb_arr, output_scores)

        dot_result = torch.from_numpy(output_scores)
        print("dot_result shape: ", dot_result.size())

    else:
        #  print("corpus_emb dtype: ", corpus_emb.dtype)
        #  print('promt emb dtype: ', prompt_emb.dtype)
         dot_result =dot_score(corpus_emb, prompt_emb)
         print(" cpu dot_result shape: ", dot_result.size())

    values, indices = torch.topk(dot_result, 4, dim=0)

    print("values: ", values)
    # print("indices: ", indices)
    rag_ele = []
    for i in range(len(values)):
        rag_ele.append(raw_text[indices[i]])
        # print("content: ", raw_text[indices[i]])
        # print("values: ", values[i])
        # print("--"*100)

    


    # del st.session_state.rag_offloader

    return rag_ele

def dot_score(a: torch.Tensor, b: torch.Tensor):
    """
    Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return torch.mm(a, b.transpose(0, 1))#torch.round(torch.mm(a, b.transpose(0, 1))*35*2/2**16/2).to(torch.int8)


# from transformers import LlamaForCausalLM, LlamaTokenizer
def check_session_state():
    black_list = ['messages', 'real_prompt', 'corpus_emb', 'raw_text', 'history']
    print("="*100)
    for key in st.session_state.keys():
        if key not in black_list:
            print("key: ", key, end="  =>  ")
            print(st.session_state[key])
    print("="*100)

def consume_output(process):
    while st.session_state.running_state:
        # print("running !!")
        try:
            char = process.stdout.read(1)#.read(1)
            if not char:
                break
            # gen_temp_img.empty()
            lottie_container.empty()
            yield char
        except:
            break

def clear_text(text):
    return text.replace("'", "'\\''")

def shellquote(s, sys_prompt, rag_element=None):

    # multi_round = "<s>[INST] <<SYS>>\n{sys}\n<</SYS>>\n\n{question1} [/INST] {model_answer_1} </s><s>[INST] {question2} [/INST]"
    question = ""
    system_setting = ""

    if sys_prompt != "":
        if st.session_state.model_name == "TAIDE" :
            # '''for taide '''
            system_setting = sys_prompt + "，只會用繁體中文回答問題，盡可能簡短地回答"
        elif st.session_state.model_name == "Llama-2-7B":
            # '''for llama '''
            system_setting = sys_prompt + ", aiming to keep your answers as brief as possible ." #, please respond as briefly as possible.
    else:
        if st.session_state.model_name == "TAIDE" :
            # '''for taide '''
            system_setting = "你是一個來自台灣的AI助理，你的名字是 TAIDE，樂於以台灣人的立場幫助使用者，只會用繁體中文回答問題，盡可能簡短地回答"
        elif st.session_state.model_name == "Llama-2-7B":
            # '''for llama '''
            system_setting = "You are a helpful AI assistant, aiming to keep your answers as brief as possible ." #, please respond as briefly as possible.


    
    # print("system setting: ", system_setting)
    question = s

    if rag_element:

        context = ""

        for i in rag_element:
            print(i)
            context = context + i + "\n"

        if st.session_state.model_name == "TAIDE" :
            system_setting = system_setting + "。使用以下上下文來回答使用者的問題。如果你不知道答案，就說你不知道，不要試圖編造答案。"
            question = "Context: " + context +"Question: " + question + "只回覆有用的答案，不回覆任何其他內容。"
        elif st.session_state.model_name == "Llama-2-7B":
            system_setting = system_setting + " Use the following context to answer the user's question. If you don't know the answer, just say that you don't know, don't try to make up an answer."
            question = "Context: " + context +"Question: " + question + "Only return the helpful answer below and nothing else."
        
    system_setting = system_setting.replace("'", "'\\''")
    question = question.replace("'", "'\\''")
    
    
    


    # memory 
    prompt_template = "<s>[INST] <<SYS>>\n"+system_setting+"\n<</SYS>>\n\n"+question+" [/INST]"
    # else:
    if len(st.session_state.history) == 0 or st.session_state.rag_on == True:
        prompt_template = "<s>[INST] <<SYS>>\n"+system_setting+"\n<</SYS>>\n\n"+question+" [/INST]"
    else:
        
       prompt_template = "<s>[INST] <<SYS>>\n"+system_setting+"\n<</SYS>>\n\n"+clear_text(st.session_state.history[0]["prompt"])+" [/INST] " \
        + clear_text(st.session_state.history[0]['content'])+" </s><s>[INST] "+question +" [/INST]"
        # prompt_template = "<s>[INST] <<SYS>>\n"+system_setting+"\n<</SYS>>\n\n"+question+" [/INST]"
    
    return prompt_template #"'"+ prompt_template + "'"

def session_state_init():
    if "model_name" not in st.session_state:
        st.session_state.model_name=""

    if "model_select" not in st.session_state:
        st.session_state.model_select="TAIDE"

    if "model_path" not in st.session_state:
        st.session_state.model_path = "../../output_weight/taide_lx_7b_chat/taide_llama2"

    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = ""

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "real_prompt" not in st.session_state:
        st.session_state.real_prompt = ""

    if "current_pid" not in st.session_state:
        st.session_state.current_pid = None
    # '''model path init '''
    # st.session_state.model_path = "../../output_weight/taide_lx_7b_chat/taide_llama2"
    if "running_state" not in st.session_state: 
        st.session_state.running_state = False
    
    if "rag_on" not in st.session_state:
        st.session_state.rag_on = False
    
    if "rag_filename" not in st.session_state:
        st.session_state.rag_filename = None
    
    if "rag_db" not in st.session_state:
        st.session_state.rag_db="customized"

    if "rag_customized_db_done" not in st.session_state:
        st.session_state.rag_customized_db_done = False
    
    if "history_counter" not in st.session_state:
        st.session_state.history_counter = 0
    
    if "history" not in st.session_state:
        st.session_state.history = []

    if "rag_device" not in st.session_state:
        st.session_state.rag_device = "cpu"

    if "rag_viper_loaded" not in st.session_state:
        st.session_state.rag_viper_loaded = False

    if "corpus_emb" not in st.session_state:
        st.session_state.corpus_emb = []

    if "raw_text" not in st.session_state:
        st.session_state.raw_text = []

    if "rag_offloader" not in st.session_state:
        st.session_state.rag_offloader = None

    # if "rag_viper_loaded" not in st.session_state:
    #     st.session_state.rag_viper_loaded = False
    
    if "llm_engine" not in st.session_state:
        st.session_state.llm_engine = "viper"
    
    if "viper_weight_downloaded" not in st.session_state:
        st.session_state.viper_weight_downloaded = False

    if "rec_prompt" not in st.session_state:
        st.session_state.rec_prompt = ""
    
    if "valid_rec_prompt" not in st.session_state:
        st.session_state.valid_rec_prompt = False

    if "example_prompt_pressed" not in st.session_state:
        st.session_state.example_prompt_pressed = ""
    
    if "running_process" not in st.session_state:
        st.session_state.running_process = None

def stop_btn_click_cb():

    if st.session_state.running_state and st.session_state.current_pid != None:
            print("befor kill PID: ", st.session_state.current_pid)
            os.killpg(os.getpgid(st.session_state.current_pid), signal.SIGUSR1)
            os.killpg(os.getpgid(st.session_state.current_pid+1), signal.SIGUSR1)
            # st.session_state.running_process.terminate()
            st.session_state.current_pid = None
            time.sleep(0.1)
            st.session_state.running_process = None
            st.session_state.running_state = False
            st.session_state.valid_rec_prompt = False
            st.session_state.example_prompt_pressed = ""

    
    stop_btn_container.empty()
    # os.killpg(os.getpgid(st.session_state.current_pid), signal.SIGUSR1)
    # os.killpg(os.getpgid(st.session_state.current_pid+1), signal.SIGUSR1)
    # st.session_state.current_pid = None
    # time.sleep(0.1)
    # st.session_state.running_state = False
    return


def clear_btn_click_cb():  
    st.session_state.messages= [] 
    st.session_state.history = []
    st.session_state.history_counter = 0 
    return

def record_btn_click_cb(recognizer, whisper_model):

    if st.session_state.running_state and st.session_state.current_pid != None:
            print("befor kill PID: ", st.session_state.current_pid)
            os.killpg(os.getpgid(st.session_state.current_pid), signal.SIGUSR1)
            os.killpg(os.getpgid(st.session_state.current_pid+1), signal.SIGUSR1)
            # st.session_state.running_process.terminate()
            st.session_state.current_pid = None
            time.sleep(0.1)
            st.session_state.running_process = None
            st.session_state.running_state = False
            st.session_state.valid_rec_prompt = False
            st.session_state.example_prompt_pressed = ""

    AUDIO_FILE = "test.wav"
    # activate mic to recording
    with sr.Microphone() as source:
        #noise calibration
        # r.adjust_for_ambient_noise(source)
        print("Pelease Say something~")
        audio = recognizer.listen(source, phrase_time_limit=15)
        print("Recording completed!")
        

    #print("type of audio object: ", type(audio))
    #save to file
    with open("./"+AUDIO_FILE, "wb") as file:
        file.write(audio.get_wav_data())
        file.close()


    print("===========================================")
    st.session_state.rec_prompt  = whisper_model.transcribe("test.wav")['text']
    # st.session_state.rec_prompt = "it's the recorded text: haha"

    # set this as the last step
    st.session_state.valid_rec_prompt = True
    return

@st.cache_resource
def loading_emb_model():
    return SentenceTransformer("WhereIsAI/UAE-Large-V1")

@st.cache_resource
def loading_whisper_model():
    return whisper.load_model("base")

@st.cache_resource
def loading_recognizer():
    return sr.Recognizer()



    

if __name__ == "__main__":

    st.markdown("<h1 style='text-align: center; color: white;'>RAG-LLM All on Viper</h1>", unsafe_allow_html=True)
    # st.title("RAG-LLM All on Viper")
    # header = st.container()
    # header.title("RAG-LLM All on Viper")
    # header.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)

    # ### Custom CSS for the sticky header
    # st.markdown(
    #     """
    # <style>
    #     div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
    #         position: sticky;
    #         top: 2.875rem;
    #         background-color: black;
    #         text-align: center;
    #         z-index: 999;
    #     }
    #     .fixed-header {
    #         border-bottom: 1px solid black;
    #         text-align: center;
    #     }
    # </style>
    #     """,
    #     unsafe_allow_html=True
    # )
        
    session_state_init()
    check_session_state()

    # params
    db_path = "../../db_folder/"
    save_path = "../../uploaded_files/"
    viper_file_path = "./rag_viper_sample_code/"
    target_viper = "neuchips_ai_epr-1"
    config = "config.csv"
    preload_ddr_data = "ddr_data.csv"
    test_data = "test_data.csv"
    # corpus_emb = None
    # raw_text = None

    embedding_model = loading_emb_model()#SentenceTransformer("WhereIsAI/UAE-Large-V1")
    whisper_model = loading_whisper_model()
    recognizer =loading_recognizer()
    recognizer.energy_threshold = 3000


    st.markdown(
        """
        <style>
            [data-testid=stSidebar] [data-testid=stImage]{
                text-align: center;
                display: block;
                margin-left: auto;
                margin-right: auto;
                width: 100%;
            }
        </style>
        """, unsafe_allow_html=True
    )


    with st.sidebar:
        
        # pick your model
        support_model_list = ['TAIDE', "Llama-2-7B"]

        st.image("./company_logo/Logo_NEUCHIPS_v_c+w.png", width = 300)

        # st.title("RAG-LLM on Viper")
        # st.write("Select your LLM")
        value = st.selectbox("Select your model", support_model_list)
        # print("last model: ", last_model)

        if value == "TAIDE" and st.session_state.model_name != "TAIDE":
            
            # st.balloons()
            
            # with st.spinner("Downloading weight to N3000 ..."):
            #     p = subprocess.check_output(" ./neu_weight_util --weight ../../output_weight/taide_lx_7b_chat/ --n3kid 0", shell=True, stderr=subprocess.STDOUT)

            #     # st.success(" TAIDE is ready !")
            st.session_state.messages = []
            st.session_state.history = []
            st.session_state.real_prompt = ""
            st.session_state.model_name = "TAIDE"
            
            # st.session_state.model_path = "../../output_weight/taide_lx_7b_chat/taide_llama2"
            
        elif value  == "Llama-2-7B" and st.session_state.model_name != "Llama-2-7B":
            
            # with st.spinner("Downloading weight to N3000 ..."):
                
            #     p = subprocess.check_output("./neu_weight_util --weight ../../output_weight/7B-chat/ --n3kid 0", shell=True, stderr=subprocess.STDOUT)
            #     # st.success("Llama-2-7B is ready !")
            st.session_state.messages = []
            st.session_state.history = []
            st.session_state.real_prompt = ""
            st.session_state.model_name = "Llama-2-7B"
            # st.session_state.model_path = "../../output_weight/7B-chat/llama2_7b_cht"

        st.session_state.model_select = value

        
        with st.container():
            # st.write("LLM Inference Engine")
            llm_engine = st.radio(
                "Select LLM Engine 👇",
                ["CPU", "Neuchips Viper"],
                key="llm_hardware",
                label_visibility="visible",
                disabled=False,
                horizontal=True,
            )

            if llm_engine == "Neuchips Viper":
                st.session_state.llm_engine = "viper"

                if st.session_state.model_name == "TAIDE":
                    st.session_state.model_path = "../../output_weight/taide_lx_7b_chat/taide_llama2"

                    if st.session_state.model_select != "TAIDE" or not st.session_state.viper_weight_downloaded :
                        with st.spinner("Downloading weight to N3000 ..."):
                            p = subprocess.check_output(" ./neu_weight_util --weight ../../output_weight/taide_lx_7b_chat/ --n3kid 0", shell=True, stderr=subprocess.STDOUT)
                            st.session_state.viper_weight_downloaded = True
                else:
                    st.session_state.model_path = "../../output_weight/7B-chat/llama2_7b_cht"

                    if st.session_state.model_select != "Llama-2-7B" or not st.session_state.viper_weight_downloaded :
                        with st.spinner("Downloading weight to N3000 ..."):
                            p = subprocess.check_output("./neu_weight_util --weight ../../output_weight/7B-chat/ --n3kid 0", shell=True, stderr=subprocess.STDOUT)
                            st.session_state.viper_weight_downloaded = True
            else:
                st.session_state.llm_engine="cpu"
                if st.session_state.model_name == "TAIDE":
                    st.session_state.model_path = "../../cpu_model/taide_7b.gguf"
                else:
                    st.session_state.model_path = "../../cpu_model/llama2_7b_chat.gguf"
            

        # with st.container():#st.form("my_form"):
        @st.dialog("Character Design")
        def char_design():
            system_prompt = st.text_area("Naming your Assistant", "")
            if st.button("Submit"):
                if system_prompt:
                    st.session_state.system_prompt = system_prompt
                st.rerun()

        # st.write("Design Your Own LLM")
        with st.container():
            st.write("Design Your Own LLM")
            if st.button("Character Design", use_container_width=True):
                char_design()



        on = st.toggle("RAG Function")

        # rag state change => clear history
        if on != st.session_state.rag_on:
            print("clear")
            st.session_state.history = []
            st.session_state.history_counter = 0

        st.session_state.rag_on = on

        with st.container(border=True):
            
        
            check_session_state()
            choice = st.radio(
                "Select Database 👇",
                ["DBpedia", "Create DB"],
                key="visibility",
                label_visibility="visible",
                disabled=not st.session_state.rag_on,
                horizontal=True,
            )
            print("choice: ", choice)
            
            if choice == "Create DB":
                st.session_state.rag_db="customized"
                uploaded_file = st.file_uploader("Choose a PDF file", accept_multiple_files=False, label_visibility="hidden")
                
                if uploaded_file is not None:
                    print("puloade_file")
                    byte_data = uploaded_file.read()
                    saved_name =uploaded_file.name
                    with open(save_path+saved_name, 'wb') as f: 
                        f.write(byte_data)
                    print('save_name: ', saved_name)
                    st.session_state.rag_filename = saved_name
                    st.session_state.rag_device = "cpu"

            elif choice == "DBpedia":
                st.session_state.rag_db="DBpedia"
                st.image("./company_logo/DBpediaLogo.png", width=150) 
            

                # device_choice = st.radio(
                #         "Select Device ",
                #         ["CPU", "Neuchips Viper"],
                #         key="device_radio",
                #         label_visibility="visible",
                #         disabled= not (True if st.session_state.rag_db == 'DBpedia' else False) or not st.session_state.rag_on,
                #         horizontal=True,
                #     )
                device_choice = "CPU"
                
                if device_choice == "Neuchips Viper":
                    # st.session_state.rag_device = "viper"

                    # set to None to disable viper device
                    devices = None #pro.get_available_device()
                    
                    # make sure viper for rag is ready
                    print("target_viper: ", target_viper)
                    if target_viper not in devices and not st.session_state.rag_viper_loaded:
                        print("Error. The device ", target_viper, " is not available.")
                        st.error("Error. The device Neuchips Viper is not available.", icon="🚨")
                        st.session_state.rag_device = "cpu"

                    else:
                        # st.success('Neuchips Viper is Ready', icon="✅")
                        st.session_state.rag_device ="viper"

                else:
                    st.session_state.rag_device = "cpu"
                

            submitted = st.button("Submit", use_container_width=True, disabled= not st.session_state.rag_on)
            if submitted:
                # st.session_state.rag_on = True
            
                
                with st.spinner("Preparing RAG Database ~"):

                    if st.session_state.rag_db == "DBpedia":
                        if st.session_state.rag_device == 'viper':

                            # st.session_state.rag_offloader = None

                            

                            
                            # org code
                            rag_offloader = pro.PyRagOffloader()
                            rag_offloader.bind_device(target_viper)
                            rag_offloader.set_config(db_path+"dbpedia/"+config)
                            rag_offloader.init_ddr(db_path+"dbpedia/"+preload_ddr_data)
                            

                            st.session_state.rag_offloader = rag_offloader
                            del rag_offloader

                            if not st.session_state.rag_offloader.is_valid():
                                st.error("Error. Can not create rag offloader.")
                                
                            st.session_state.rag_viper_loaded = True

                            # preload DB in to Viper
                        else:
                            # loading corpus
                            corpus_emb = torch.load(db_path+"dbpedia/dbpedia_corpus_emb.pt", map_location=torch.device('cpu') ).to(torch.float32)
                            
                            #loading raw text
                        
                        raw_text = torch.load(db_path+"dbpedia/dbpedia_raw_text.pt", map_location=torch.device('cpu') )#.to(torch.float16)

                    elif st.session_state.rag_db == "customized":
                        basename = os.path.splitext(st.session_state.rag_filename)[0]
                        save_db_name = basename +".pt"

                        if os.path.isfile(db_path+basename+'_corpus_emb.pt'):
                            corpus_emb = torch.load(db_path+basename+"_corpus_emb.pt")
                            raw_text = torch.load(db_path+basename+"_raw_text.pt")
                        else:
                            pdf_start = time.time()
                            raw_pdf_elements = partition_pdf(
                                filename= save_path + st.session_state.rag_filename,
                                # Unstructured first finds embedded image blocks
                                extract_images_in_pdf=False,
                                # in
                                # Titles are any sub-section of the document
                                infer_table_structure=True,
                                # Post processing to aggregate text once we have the title
                                chunking_strategy="by_title",
                                # Chunking params to aggregate text blocks
                                # Attempt to create a new chunk 3800 chars
                                # Attempt to keep chunks > 2000 chars
                                max_characters=400,
                                new_after_n_chars=380,
                                combine_text_under_n_chars=200,
                                image_output_dir_path="./parsing_data",
                            )
                            print("pdf parsing time: ", time.time()-pdf_start)
                            

                            # Create a dictionary to store counts of each type
                            category_counts = {}

                            

                            class Element(BaseModel):
                                type: str
                                text: Any


                            # Categorize by type
                            categorized_elements = []
                            for element in raw_pdf_elements:
                                if "unstructured.documents.elements.Table" in str(type(element)):
                                    categorized_elements.append(Element(type="table", text=str(element)))
                                elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
                                    categorized_elements.append(Element(type="text", text=str(element)))

                            # Tables
                            table_elements = [e for e in categorized_elements if e.type == "table"]
                            
                            text_elements = [e for e in categorized_elements if e.type == "text"]
                        

                            raw_text = []

                            corpus_start = time.time()
                            for i in tqdm(range(len(text_elements))):
                                raw_text.append(text_elements[i].text)

                            for t in tqdm(range(len(table_elements))):
                                raw_text.append(table_elements[t].text)

                            # embedding_model = SentenceTransformer("WhereIsAI/UAE-Large-V1")
                            corpus_emb = embedding_model.encode(raw_text, convert_to_tensor=True)
                            torch.save(corpus_emb, db_path+basename+"_corpus_emb.pt")
                            torch.save(raw_text, db_path+basename+"_raw_text.pt")
                            # st.session_state.rag_customized_db_done = True
                            print(corpus_emb.size())
                            # print(Find_scale(corpus_emb))
                            print("get corpus db time: ", time.time() - corpus_start)

                    if st.session_state.rag_device != "viper":
                        st.session_state.corpus_emb = corpus_emb
                    
                    st.session_state.raw_text = raw_text
                    st.success('RAG Database is Ready', icon="✅")
             

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar= "❇️"if message["role"] == "assistant" else "🧙‍♀️"):
            st.markdown(message["content"])


    button_cols = _bottom.columns(3)

    if (st.session_state.rag_db != "customized" and st.session_state.rag_on) or not st.session_state.rag_on:
        example_prompts = example_generator()
        

        if button_cols[0].button(example_prompts[0], use_container_width=True):
            st.session_state.example_prompt_pressed = example_prompts[0]


        if button_cols[1].button(example_prompts[1], use_container_width=True):
            if example_prompts[1] == "請幫我總結台版晶片法案":
                st.session_state.example_prompt_pressed  = "請將這篇文章精簡條理化:「產業創新條例第10條之2及第72條條文修正案」俗稱「台版晶片法」,針對半導體、電動車、5G等技術創新且居國際供應鏈關鍵地位公司,提供最高25%營所稅投抵優惠,企業適用要件包含當年度研發費用、研發密度達一定規模,且有效稅率達一定比率。\
為因應經濟合作暨發展組織(OECD)國家最低稅負制調整,其中有效稅率門檻,民國112年訂為12%,113年料將提高至15%,但仍得審酌國際間最低稅負制實施情形。\
經濟部官員表示,已和財政部協商進入最後階段,除企業研發密度訂在6%,目前已確認,企業購置先進製程的設備投資金額達100億元以上可抵減。\
財政部官員表示,研商過程中,針對台灣產業與其在國際間類似的公司進行深入研究,在設備部分,畢竟適用產創10之2的業者是代表台灣隊打「國際盃」,投入金額不達100億元,可能也打不了。\
至於備受關注的研發費用門檻,經濟部官員表示,歷經與財政部來回密切討論,研發費用門檻有望落在60億至70億元之間。\
財政部官員指出,研發攸關台灣未來經濟成長動能,門檻不能「高不可攀」,起初雖設定在100億元,之所以會調降,正是盼讓企業覺得有辦法達得到門檻、進而適用租稅優惠,才有動力繼續投入研發,維持國際供應鏈關鍵地位。\
經濟部官員表示,因廠商研發費用平均為30、40億元,其中,IC設計業者介於30億至60億元範圍,若將門檻訂在100億元,符合條件的業者較少、刺激誘因不足;此外,若符合申請門檻的業者增加,將可提高企業在台投資金額,財政部稅收也能因此獲得挹注。\
IC設計業者近日頻頻針對產創10之2發聲,希望降低適用門檻,加上各國力拚供應鏈自主化、加碼補助半導體產業,經濟部官員表示,經濟部和財政部就產創10之2達成共識,爭取讓更多業者受惠,盼增強企業投資力道及鞏固台灣技術地位。\
財政部官員表示,租稅獎勵的制定必須「有為有守」,並以達到獎勵設置目的為最高原則,現階段在打「國內盃」的企業仍可適用產創第10條、10之1的租稅優惠,共同壯大台灣經濟發展。\
經濟部和財政部正就研發費用門檻做最後確認,待今明兩天預告子法之後,約有30天時間,可與業界進一步討論及調整,盼產創10之2能在6月上路。"
            else:
                st.session_state.example_prompt_pressed  = example_prompts[1]

        if button_cols[2].button(example_prompts[2], use_container_width=True):
            st.session_state.example_prompt_pressed = example_prompts[2]

    cols = _bottom.columns([1.9, 17.2, 2.5, 2.6], gap="small")

    with cols[1]:
        # st.session_state._pprompt = st.chat_input("What is up?")
        # print("prompt: ", st.session_state._pprompt)
        prompt = st.chat_input("What is up?", key="chat_in")
        

    
    with cols[0]:
       record_btn = st.button(":studio_microphone:", use_container_width=True)


    if record_btn:

        record_msg_container= st.empty()
        with record_msg_container:
            st.markdown("recording~")
            record_btn_click_cb(recognizer, whisper_model)
            record_msg_container.empty()

    # for whisper
    if st.session_state.valid_rec_prompt:
        prompt = st.session_state.rec_prompt

    if st.session_state.example_prompt_pressed !="":
        prompt = st.session_state.example_prompt_pressed
    
    print("prompt: ", prompt)
    

    if prompt: #:= st.chat_input("What is up?"):
        # check_session_state()
        

        if st.session_state.running_state and st.session_state.current_pid != None:
            print("befor kill PID: ", st.session_state.current_pid)
            os.killpg(os.getpgid(st.session_state.current_pid), signal.SIGUSR1)
            os.killpg(os.getpgid(st.session_state.current_pid+1), signal.SIGUSR1)
            # st.session_state.running_process.terminate()
            st.session_state.current_pid = None
            time.sleep(0.1)
            st.session_state.running_process = None
            st.session_state.running_state = False
            st.session_state.valid_rec_prompt = False
            st.session_state.example_prompt_pressed = ""
            


        # RAG mlcommon
        if st.session_state.rag_on:
            print("rag on !")
            rag_element = rag_process(prompt, st.session_state.corpus_emb, st.session_state.raw_text, embedding_model)
            print("rag process done !")
            # st.session_state.rag_offloader = None
            # for i in rag_element:
            #     print(i)
            #     print("--"*100)
            word_esc = shellquote(prompt, st.session_state.system_prompt, rag_element)
        else:
            word_esc = shellquote(prompt, st.session_state.system_prompt)

        # print("word_esc: ", word_esc)
        


    
        while st.session_state.running_state == False:

            if st.session_state.llm_engine == 'viper':
                process = subprocess.Popen("./main --n3k_id %d -m %s -c 4096 -p %s -n %d -b 32 --temp 0 --no-display-prompt" % (0, st.session_state.model_path, "'"+word_esc+"'", 512), stdout=subprocess.PIPE, text=True, shell=True, close_fds=True ,preexec_fn = os.setsid)
            else:
                process = subprocess.Popen("./cpu_main -m %s -c 4096 -p %s -n %d -b 32 --temp 0 --prompt-print-skip" % (st.session_state.model_path, "'"+word_esc+"'", 512), stdout=subprocess.PIPE, text=True, shell=True, close_fds=True ,preexec_fn = os.setsid)
                
            # st.session_state.running_process = process
            st.session_state.running_state = True
            st.session_state.current_pid = process.pid

            
            
            with cols[2]:
                stop_btn_container = st.empty()
                with stop_btn_container:
                    stop_btn = st.button("Stop", on_click=stop_btn_click_cb, key=f"abc", use_container_width=True)

            
            print("current process id: ", process.pid)

    
            with st.chat_message("user", avatar="🧙‍♀️"):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant", avatar="❇️"):
                # print("Before write stream runnning state check: ", st.session_state.running_state)
                lottie_container = st.empty()

                    # gen_temp_img = st.image("./company_logo/Logo_NEUCHIPS_h_c+w.png", width=300)
                    # with st.echo():
                    # gen_temp_img = st_lottie("./UI_elements/loading.json", height=200, width=300)
                with lottie_container:
                    with open("./UI_elements/processing_animation.json", "r") as f:
                        data = json.load(f)

                    st.markdown("""
                            <style>
                                iframe {
                                    justify-content: center;
                                    display: block;
                                    border-style:none;
                                }
                            </style>
                            """, unsafe_allow_html=True
                        )
                    st_lottie(data, height=80, key='gen')
                    # st_lottie("./processing.json")
                
                    response = st.write_stream(consume_output(process))
                    stop_btn_container.empty()
                
                print("After write stream runnning state check: ", st.session_state.running_state)
                
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": response})
            
                    

            
            print("before history counter")
            check_session_state()
            st.session_state.history_counter += 1
            
            if st.session_state.history_counter < 2:
                st.session_state.history.append({"prompt": prompt, "content": response})
            else:
                st.session_state.history.append({"prompt": prompt, "content": response})
                print("st.session_state.history len: ", len(st.session_state.history))
                del st.session_state.history[0]
                print("after del first st.session_state.history len: ", len(st.session_state.history))
                st.session_state.history_counter = 1

            print("after history counter ")
            check_session_state()

            # st.session_state.real_prompt = word_esc + clear_text(response) + "</s>"
            
            # st.session_state.running_state = False
        
        st.session_state.current_pid = None
        st.session_state.running_state = False
        st.session_state.valid_rec_prompt = False
        st.session_state.example_prompt_pressed=""

        if len(st.session_state.messages) != 0:
            with cols[3]:
                st.button("clear", on_click=clear_btn_click_cb, use_container_width=True)

        
        # print("call empty()")
        # placeholder.empty()
