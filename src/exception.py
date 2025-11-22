import sys
import logging

def error_message(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    return (
        f"Error occurred in python script [{file_name}] "
        f"line number [{exc_tb.tb_lineno}] "
        f"error message [{str(error)}]"
    )


class CustomException(Exception):
    def __init__(self, error, error_detail: sys):
        # generate detailed error message
        message = error_message(error, error_detail)
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message


if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        logging.info("Divide by zero")
        raise CustomException(e, sys)
