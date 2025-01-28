import sys
import logging

def error_message_details(error,error_detail:sys):
    
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_msg = f"error occurred in [{file_name}] at line number [{exc_tb.tb_lineno}] with error message [{str(error)}]"
    
    return error_msg

    
class Custom_Exception(Exception):
    def __init__(self,error_msg,error_detail:sys):
        super().__init__(error_msg)
        self.error_msg = error_message_details(error_msg, error_detail = error_detail)
    
def __str__(self):
    return self.error_msg


    