import os
import pymupdf4llm
from transformers import AutoTokenizer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader


model = 'meta-llama/Meta-Llama-3-70B-Instruct'

tokenizer = AutoTokenizer.from_pretrained(model)

class Agent:
    def __init__(self, client, name, db_path):
        self.client = client
        self.name = name
        self.persona = "" 
        self.memory = ""
        self.conversation_history = " " 
        self.path = f"{db_path}/{self.name}_chroma_db"
        
        if os.path.exists(db_path):
            # Load existing database
            self.db = Chroma(persist_directory=self.path, embedding_function=HuggingFaceEmbeddings())
        else:
            # Create a new database
            self.db = Chroma(persist_directory=self.path, embedding_function=HuggingFaceEmbeddings())
            self.db.persist()
    
    def _get_documents(self, query, k = 5):
        if(self.db == None):
            return []
        
        docs = self.db.similarity_search(query, k)

        return [doc.page_content for doc in docs]
    
    def _generate_response(self, chat, max_tokens):
        messages = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        
        text = ''
        for response in self.client.generate_stream(
            messages,
            stop_sequences= ['<|eot_id|>'],
            top_p=0.9,
            do_sample=True,
            temperature=1,
            max_new_tokens=max_tokens,
            return_full_text = False
        ):
            if not response.token.special:
                text += response.token.text
            print(response.token.text, end='', flush=True)
        
        return response.generated_text.strip()
        
    def _update_memory(self):
        chat = [
            {"role": "assistant", "content": f"""{self.persona}
             Welcome to your Memory bank. You will be given your most recent conversation. Based on the conversation you had, write organized notes that you would like to recall under the assumption that your won't remember this information in the future. Your notes should be concise and relevant to the context, your role, and who you are. When taking notes, ensure that they can be useful to your future self."""},
            {"role": "user", "content": f"Here is the conversation history: {self.conversation_history}"}
        ]
        response = self._generate_response(chat, 128)
        self.memory += "\n" + response
        
    def _get_memory(self):
        temp_string = f"This is your memory. Your memory contains information that you previously decided to keep to help you right now. Do not mention anything about your memory. Just use your memory to respond to instructions when necessary. Your Memory: {self.memory}. "
        return temp_string
    
    def _get_persona(self, instruction):
        chat = [
            {"role": "assistant", "content": f"""You will be given a user instruction.

            Generate a prompt that provides a role generally relevant to the given instruction without being too specific.

            For example: You are an expert in ___ with a specialization in ___.

            Only provide the new prompt."""},
            {"role": "user", "content": f"User Instruction: {instruction}"}
        ]
        response = self._generate_response(chat, 32)
        return response
    
    def _update_conv(self, conv):
        self.conversation_history += "\n" + conv
        
    def instruct(self, instruction):
        self._update_conv(f"User: {instruction}")
        
        docs = self._get_documents(instruction, k = 5)
        
        chat = [
            {"role": "assistant", "content": f"""Your name is {self.name}. You are to keep all your responses concise. Respond to the instruction. You will also be provided your memory and your conversation history.
             
            Memory: {self._get_memory()}
            
            Conversation history: {self.conversation_history}
             
            You are a helpful Chatbot. You will be given a User Instruction to respond to. Here is some information that may help you with your response : {docs}"""},            
            {"role": "user", "content": f"User Instruction: {instruction}."}
        ]
        
        response = self._generate_response(chat, 256)
        self._update_conv(f"{self.name}: " + response)
        self._update_memory()
        return f"{self.name}: " + response
    
    # converts pdf to txt, embeds, and adds to db
    def add_to_db(self, pdf_document_path):
        md_text = pymupdf4llm.to_markdown(pdf_document_path)
        
        temp_file_path = 'TEMP_extracted_text.txt'
        with open(temp_file_path, 'w') as file:
            file.write(md_text)

        loader = TextLoader(temp_file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        c_db = Chroma(persist_directory=self.path, embedding_function=HuggingFaceEmbeddings())
        c_db.add_documents(splits)
        c_db.persist()
        print("File Successfully Uploaded.")